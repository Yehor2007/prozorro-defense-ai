"""
update_data.py
Incremental ETL for the Prozorro Defense AI Explorer — the nightly-safe
counterpart to extract_prozorro.py's from-scratch historical backfill.
Invoked by telegram_bot.py's APScheduler job (3:00 AM daily) and by the
crontab example in DEPLOY.md; can also be run manually.

Design:
1. Resume point = MAX(date_modified) already in DuckDB, not a fixed date
   — every run picks up exactly where the last one left off. Verified
   live against the real API: passing that exact timestamp as the
   `offset` param returns only tenders modified strictly after it.
2. Unlike extract_prozorro.py's default backfill mode (which skips a
   tender_id whose raw JSON already exists locally), every tender in this
   incremental window gets its raw JSON re-fetched and overwritten
   (extract_prozorro.run_extraction(..., overwrite=True)) — being in the
   window at all means it's either brand new or was modified since our
   last sync, so any on-disk copy could be stale.
3. Merge into DuckDB by concatenating the existing tables with the
   freshly flattened new/updated rows and re-running
   transform_prozorro.finalize_dataframes() over the combined whole —
   that function already drops duplicate tender_ids keeping the most
   recently modified version (it was written for exactly this "tender
   re-fetched across separate sync runs" case), and recomputes
   price_deviation_pct correctly across the full table (a new tender
   landing in a CPV group can shift the group median for every tender in
   it, not just itself). The API fetch — the genuinely slow, rate-limited
   part — stays incremental; this local merge is cheap even as a full
   recompute, since it only touches already-parsed, already-small data
   (no re-fetching, no re-parsing thousands of raw JSON files).
4. Re-embed and upsert ONLY the touched tenders into ChromaDB —
   collection.upsert() is idempotent by ID (see load_chroma.py), so this
   never creates duplicates or needs a delete step.

If DuckDB doesn't exist yet or has no tenders, this refuses to run an
unbounded from-2015 crawl by accident — run extract_prozorro.py's
historical backfill first.

Usage:
    python update_data.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import chromadb
import duckdb
import pandas as pd

import extract_prozorro
import load_chroma
import transform_prozorro

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prozorro_update")

DB_PATH = Path("data/prozorro.duckdb")
CHROMA_PATH = Path("chroma_db")
RAW_DIR = extract_prozorro.RAW_DIR

# Safety cap for a single incremental run — a nightly sync should never
# need anywhere near this many; if it does, something upstream (e.g. a
# long outage) needs a human to look, not an unbounded catch-up crawl.
MAX_TENDERS_PER_RUN = 2000


def get_resume_cursor(db_path: Path) -> str | None:
    """ISO timestamp to resume the incremental fetch from, or None if
    there's nothing to resume from (DB missing / no tenders yet)."""
    if not db_path.exists():
        return None
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute("SELECT MAX(date_modified) FROM tenders").fetchone()
    finally:
        con.close()
    if not row or row[0] is None:
        return None
    return row[0].isoformat()


def hours_since_last_sync(db_path: Path) -> float | None:
    """None if there's no baseline yet (see get_resume_cursor) — used by
    telegram_bot.py's startup catch-up check: cron-style schedulers
    (APScheduler included) silently skip a run if the process/VM was
    paused past their misfire grace window (confirmed live: a Docker
    Desktop host sleep on macOS caused exactly this, with zero log trace
    of the miss) — a data-freshness check at startup is what actually
    catches that, not a more generous grace period alone."""
    cursor = get_resume_cursor(db_path)
    if cursor is None:
        return None
    last = datetime.fromisoformat(cursor)
    now = datetime.now(last.tzinfo)
    return (now - last).total_seconds() / 3600


def flatten_tender_ids(tender_ids: list[str]) -> tuple[list[dict], list[dict]]:
    tender_rows, item_rows = [], []
    for tid in tender_ids:
        path = RAW_DIR / f"{tid}.json"
        if not path.exists():
            logger.warning("Expected raw JSON for %s but it's missing — skipping", tid)
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            tender_rows.append(transform_prozorro.flatten_tender(raw, path.name))
            item_rows.extend(transform_prozorro.flatten_items(raw, path.name))
        except Exception as e:
            logger.error("Failed to flatten %s: %s", tid, e)
    return tender_rows, item_rows


def merge_into_duckdb(db_path: Path, tender_ids: list[str]) -> int:
    """Merge the given (new or updated) tender_ids into DuckDB. Returns
    the number of tender rows actually merged (0 if nothing flattened)."""
    tender_rows, item_rows = flatten_tender_ids(tender_ids)
    if not tender_rows:
        return 0

    new_tenders = pd.DataFrame(tender_rows)
    new_items = pd.DataFrame(item_rows)

    con = duckdb.connect(str(db_path))
    try:
        existing_tenders = con.execute("SELECT * FROM tenders").fetchdf()
        existing_items = con.execute("SELECT * FROM items").fetchdf()

        merged_tenders = pd.concat([existing_tenders, new_tenders], ignore_index=True)
        merged_items = pd.concat([existing_items, new_items], ignore_index=True)
        merged_tenders, merged_items = transform_prozorro.finalize_dataframes(merged_tenders, merged_items)

        con.execute("CREATE OR REPLACE TABLE tenders AS SELECT * FROM merged_tenders")
        con.execute("CREATE OR REPLACE TABLE items AS SELECT * FROM merged_items")
    finally:
        con.close()

    logger.info(
        "Merged %d touched tenders — table now has %d tenders / %d items",
        len(new_tenders), len(merged_tenders), len(merged_items),
    )
    return len(new_tenders)


def reembed_touched(db_path: Path, chroma_path: Path, tender_ids: list[str]) -> None:
    df = load_chroma.load_tenders_by_ids(db_path, tender_ids)
    if df.empty:
        logger.warning("Nothing to re-embed for the touched tender_ids — skipping Chroma step")
        return

    embedder = load_chroma.Embedder(load_chroma.DEFAULT_MODEL)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=load_chroma.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    load_chroma.embed_and_upsert(df, collection, embedder)
    logger.info("Re-embedded %d tenders; collection now has %d items", len(df), collection.count())


def run_update() -> dict:
    """Returns a small summary dict — used by both the CLI entry point and
    telegram_bot.py's scheduled job (so a failed/empty run can be logged
    without needing to parse stdout)."""
    since = get_resume_cursor(DB_PATH)
    if since is None:
        msg = (
            f"{DB_PATH} does not exist or has no tenders yet — refusing to run an "
            "unbounded historical crawl. Run extract_prozorro.py's backfill first."
        )
        logger.error(msg)
        return {"status": "no_baseline", "message": msg}

    logger.info("Resuming incremental fetch from date_modified > %s", since)
    tender_ids = extract_prozorro.run_extraction(since=since, limit=MAX_TENDERS_PER_RUN, overwrite=True)

    if not tender_ids:
        logger.info("No new or updated defense-relevant tenders since last sync.")
        return {"status": "up_to_date", "since": since, "n_tenders": 0}

    n_merged = merge_into_duckdb(DB_PATH, tender_ids)
    if n_merged:
        reembed_touched(DB_PATH, CHROMA_PATH, tender_ids)

    logger.info("Incremental update complete: %d tenders touched.", n_merged)
    return {"status": "ok", "since": since, "n_tenders": n_merged}


if __name__ == "__main__":
    result = run_update()
    if result["status"] == "no_baseline":
        sys.exit(1)

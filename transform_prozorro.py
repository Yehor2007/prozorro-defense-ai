"""
transform_prozorro.py
Step 2 of the Prozorro Defense AI Explorer ETL pipeline.

Reads raw tender JSON dumped by the extraction step, flattens each tender
into a `tenders` row and its line items into `items` rows, and persists
both as tables in a local DuckDB file.

Usage:
    python transform_prozorro.py --raw-dir data/raw --db-path data/prozorro.duckdb
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prozorro_transform")


# --------------------------------------------------------------------------
# Safe accessors — real Prozorro JSON is inconsistent: fields can be absent,
# null, empty dicts, or empty lists depending on tender status/type/stage.
# (e.g. a tender still in "active.tendering" has no awards/bids/contracts
# at all — see the sample DJI drone tender used to build this script.)
# --------------------------------------------------------------------------

def safe_get(d: Optional[dict], *keys, default=None):
    """Chain .get() calls safely through nested dicts that might be None."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Flattening logic
# --------------------------------------------------------------------------

def pick_active_award(tender: dict) -> Optional[dict]:
    """
    Return the most relevant award: prefer status == 'active', otherwise
    fall back to the most recent award, otherwise None — many tenders
    (e.g. anything still in 'active.tendering') have no awards yet.
    """
    awards = tender.get("awards") or []
    if not awards:
        return None
    active = [a for a in awards if a.get("status") == "active"]
    return active[-1] if active else awards[-1]


def pick_active_contract(tender: dict) -> Optional[dict]:
    contracts = tender.get("contracts") or []
    if not contracts:
        return None
    active = [c for c in contracts if c.get("status") == "active"]
    return active[-1] if active else contracts[-1]


def flatten_tender(tender: dict, source_file: str) -> dict:
    value = tender.get("value") or {}
    buyer = tender.get("procuringEntity") or {}
    buyer_identifier = buyer.get("identifier") or {}
    buyer_address = buyer.get("address") or {}

    award = pick_active_award(tender)
    contract = pick_active_contract(tender)

    # "How real is this number" ordering: a signed contract value beats a
    # proposed award value beats the tender's own estimated budget. Any of
    # these can be missing depending on tender stage.
    awarded_amount = None
    awarded_currency = value.get("currency")
    if contract is not None:
        c_value = contract.get("value") or {}
        awarded_amount = to_float(c_value.get("amount"))
        awarded_currency = c_value.get("currency") or awarded_currency
    if awarded_amount is None and award is not None:
        a_value = award.get("value") or {}
        awarded_amount = to_float(a_value.get("amount"))
        awarded_currency = a_value.get("currency") or awarded_currency

    supplier_name, supplier_edrpou = None, None
    if award is not None:
        suppliers = award.get("suppliers") or []
        if suppliers:
            supplier_name = suppliers[0].get("name")
            supplier_edrpou = safe_get(suppliers[0], "identifier", "id")

    items = tender.get("items") or []
    first_classification = safe_get(items[0] if items else {}, "classification", default={}) or {}

    return {
        "tender_id": tender.get("id"),
        "tender_number": tender.get("tenderID"),
        "title": tender.get("title"),
        "status": tender.get("status"),
        "procurement_method": tender.get("procurementMethod"),
        "procurement_method_type": tender.get("procurementMethodType"),
        "main_procurement_category": tender.get("mainProcurementCategory"),
        "date_created": tender.get("dateCreated"),
        "date_modified": tender.get("dateModified"),
        "notice_publication_date": tender.get("noticePublicationDate"),
        "cpv_main": first_classification.get("id"),
        "cpv_description": first_classification.get("description"),
        "n_items": len(items),
        "budget_amount": to_float(value.get("amount")),
        "budget_currency": value.get("currency"),
        "vat_included": value.get("valueAddedTaxIncluded"),
        "awarded_amount": awarded_amount,
        "awarded_currency": awarded_currency,
        "n_bids": len(tender.get("bids") or []),
        "n_awards": len(tender.get("awards") or []),
        "n_contracts": len(tender.get("contracts") or []),
        "supplier_name": supplier_name,
        "supplier_edrpou": supplier_edrpou,
        "buyer_name": buyer.get("name"),
        "buyer_edrpou": buyer_identifier.get("id"),
        "buyer_kind": buyer.get("kind"),  # e.g. "defense" — cleaner relevance signal than CPV/keyword guessing
        "buyer_region": buyer_address.get("region"),
        "buyer_country": buyer_address.get("countryName"),
        "source_file": source_file,
    }


def flatten_items(tender: dict, source_file: str) -> list[dict]:
    tender_id = tender.get("id")
    rows = []
    for item in tender.get("items") or []:
        classification = item.get("classification") or {}
        unit = item.get("unit") or {}
        unit_value = unit.get("value") or {}  # frequently absent — Prozorro doesn't require per-unit price at tender stage
        delivery_address = item.get("deliveryAddress") or {}
        delivery_date = item.get("deliveryDate") or {}

        quantity = to_float(item.get("quantity"))
        unit_amount = to_float(unit_value.get("amount"))

        rows.append({
            "tender_id": tender_id,
            "item_id": item.get("id"),
            "description": item.get("description"),
            "cpv": classification.get("id"),
            "cpv_description": classification.get("description"),
            "quantity": quantity,
            "unit_name": unit.get("name"),
            "unit_code": unit.get("code"),
            "unit_amount": unit_amount,        # per-unit price, only when Prozorro discloses it
            "unit_currency": unit_value.get("currency"),
            # Never silently default missing values to 0 — that would quietly
            # corrupt downstream SUM/AVG queries. Leave as None instead.
            "line_total": (unit_amount * quantity) if (unit_amount is not None and quantity is not None) else None,
            "delivery_region": delivery_address.get("region"),
            "delivery_end_date": delivery_date.get("endDate"),
            "source_file": source_file,
        })
    return rows


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def load_raw_tenders(raw_dir: Path):
    """Yield (tender_dict, filename) pairs, skipping unparseable files."""
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        logger.warning("No JSON files found in %s", raw_dir)

    for path in files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error("Failed to parse %s: %s — skipping", path.name, e)
            continue

        # Some dumps wrap the tender body in {"data": {...}} (raw API
        # response shape), others store the tender dict directly (as saved
        # by the extraction step). Handle both.
        tender = raw.get("data") if isinstance(raw, dict) and "data" in raw and "id" not in raw else raw

        if not isinstance(tender, dict) or "id" not in tender:
            logger.warning("Unexpected structure in %s — skipping", path.name)
            continue

        yield tender, path.name


def finalize_dataframes(df_tenders: pd.DataFrame, df_items: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """dtype cleanup, dedup, and derived-column computation — split out
    from build_dataframes() so update_data.py's incremental merge can
    reuse the exact same logic on a concatenation of existing + freshly
    flattened rows, rather than re-deriving it. Safe to call with a mix
    of already-typed (e.g. read back from DuckDB) and raw string date
    columns — pd.to_datetime handles both in the same column."""
    if df_tenders.empty:
        return df_tenders, df_items

    # --- dtype cleanup ---
    for col in ("date_created", "date_modified", "notice_publication_date"):
        df_tenders[col] = pd.to_datetime(df_tenders[col], errors="coerce", utc=True)

    for col in ("budget_amount", "awarded_amount"):
        df_tenders[col] = pd.to_numeric(df_tenders[col], errors="coerce")

    for col in ("n_items", "n_bids", "n_awards", "n_contracts"):
        df_tenders[col] = pd.to_numeric(df_tenders[col], errors="coerce").astype("Int64")

    # Drop exact duplicate tender_ids (can happen if the same tender was
    # re-fetched across separate incremental-sync runs into separate files,
    # or — for update_data.py's merge — appears in both the existing table
    # and the freshly re-fetched/updated batch). Keep the most recently
    # modified version.
    before = len(df_tenders)
    df_tenders = df_tenders.sort_values("date_modified").drop_duplicates(subset="tender_id", keep="last")
    if len(df_tenders) < before:
        logger.info("Dropped %d duplicate tender rows", before - len(df_tenders))

    # --- derived analytics column: deviation from CPV-group median award ---
    # Only computed where we actually have an awarded_amount AND at least
    # 2 comparable tenders in the same CPV group — otherwise left as NaN
    # rather than a misleading 0% deviation. Recomputed over the FULL
    # table every time (not just new rows), since one new tender landing
    # in a CPV group can shift that group's median for every tender in it.
    group_median = df_tenders.groupby("cpv_main")["awarded_amount"].transform("median")
    group_size = df_tenders.groupby("cpv_main")["awarded_amount"].transform("count")
    deviation = (df_tenders["awarded_amount"] - group_median) / group_median * 100
    df_tenders["price_deviation_pct"] = deviation.where(group_size >= 2)

    if not df_items.empty:
        for col in ("quantity", "unit_amount", "line_total"):
            df_items[col] = pd.to_numeric(df_items[col], errors="coerce")
        df_items["delivery_end_date"] = pd.to_datetime(df_items["delivery_end_date"], errors="coerce", utc=True)
        df_items = df_items.drop_duplicates(subset=["tender_id", "item_id"], keep="last")

    return df_tenders, df_items


def build_dataframes(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tender_rows, item_rows = [], []
    n_ok, n_failed = 0, 0

    for tender, fname in load_raw_tenders(raw_dir):
        try:
            tender_rows.append(flatten_tender(tender, fname))
            item_rows.extend(flatten_items(tender, fname))
            n_ok += 1
        except Exception as e:
            # Never let one malformed tender kill the whole batch.
            logger.error("Failed to flatten %s: %s", fname, e)
            n_failed += 1

    logger.info("Flattened %d tenders (%d failed) from %s", n_ok, n_failed, raw_dir)

    df_tenders = pd.DataFrame(tender_rows)
    df_items = pd.DataFrame(item_rows)

    if df_tenders.empty:
        logger.warning("No tenders produced — resulting DataFrame is empty")
        return df_tenders, df_items

    return finalize_dataframes(df_tenders, df_items)


def persist_to_duckdb(df_tenders: pd.DataFrame, df_items: pd.DataFrame, db_path: Path):
    con = duckdb.connect(str(db_path))
    con.execute("CREATE OR REPLACE TABLE tenders AS SELECT * FROM df_tenders")
    con.execute("CREATE OR REPLACE TABLE items AS SELECT * FROM df_items")

    # Sanity check: catches ETL bugs early (e.g. a dedup step that didn't work).
    dup_check = con.execute(
        "SELECT tender_id, COUNT(*) c FROM tenders GROUP BY tender_id HAVING c > 1"
    ).fetchdf()
    if not dup_check.empty:
        logger.warning("Duplicate tender_id values remain in tenders table: %d", len(dup_check))

    n_tenders = con.execute("SELECT COUNT(*) FROM tenders").fetchone()[0]
    n_items = con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    logger.info("Persisted %d tenders and %d items to %s", n_tenders, n_items, db_path)
    con.close()


def main():
    parser = argparse.ArgumentParser(description="Flatten raw Prozorro tender JSON into DuckDB tables.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--db-path", type=Path, default=Path("data/prozorro.duckdb"))
    args = parser.parse_args()

    if not args.raw_dir.exists():
        logger.error("Raw directory %s does not exist", args.raw_dir)
        sys.exit(1)

    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    df_tenders, df_items = build_dataframes(args.raw_dir)

    if df_tenders.empty:
        logger.error("No tenders to persist — aborting")
        sys.exit(1)

    persist_to_duckdb(df_tenders, df_items, args.db_path)


if __name__ == "__main__":
    main()

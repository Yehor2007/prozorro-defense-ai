import requests
import time
import json
from pathlib import Path
from datetime import datetime, timezone

BASE_URL = "https://public-api.prozorro.gov.ua/api/2.5"
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# CPV prefixes we care about (defense-adjacent, non-classified)
TARGET_CPV_PREFIXES = (
    "35",       # security/defense/military equipment
    "18",       # clothing (tactical/protective)
    "34711",    # occasionally misfiled drone-adjacent items
    "33",       # medical (we'll further filter by keyword)
)

# Ukrainian + English keywords to catch items CPV misses (esp. drones)
KEYWORDS = [
    "дрон", "бпла", "квадрокоптер", "безпілотн",
    "тактичн", "військов", "бронежилет", "каска",
    "турнікет", "generator", "генератор",
]


def _get_with_retry(url: str, **kwargs) -> requests.Response:
    """A transient network error (timeout, connection reset, DNS hiccup —
    requests.exceptions.RequestException, not just an HTTP error status)
    used to propagate straight out of both call sites below and crash the
    whole run. That's tolerable interactively, but not for an unattended
    nightly job (see update_data.py) — one bad request from a multi-hour
    walk shouldn't lose all the progress made so far. One retry after a
    short backoff, then let the caller's own except clause decide what to
    do (skip this tender, or give up)."""
    try:
        resp = requests.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException as e:
        print(f"request to {url} failed ({e}), retrying once...")
        time.sleep(2)
        resp = requests.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp


def fetch_tender_ids(since: str | None = None, descending: bool = False):
    """
    Walk the /tenders list endpoint using offset-based pagination.
    `since` is an ISO date string to resume from (incremental sync) — only
    meaningful in ascending (default) mode; combining it with `descending`
    is not supported by the API's offset semantics, so it's ignored there.
    `descending=True` walks from the most recently modified tender backward
    in time (confirmed against the live API: `descending=1` returns
    today's tenders first, and next_page carries the flag forward).
    """
    params = {}
    if descending:
        params["descending"] = "1"
    elif since:
        params["offset"] = since  # Prozorro accepts a date offset directly

    while True:
        resp = _get_with_retry(f"{BASE_URL}/tenders", params=params)
        payload = resp.json()

        data = payload.get("data", [])
        if not data:
            break

        for item in data:
            yield item  # {"id":..., "dateModified":..., "status":...}

        next_offset = payload.get("next_page", {}).get("offset")
        if not next_offset or next_offset == params.get("offset"):
            break
        params["offset"] = next_offset
        time.sleep(0.15)  # be polite to the public API — no documented hard rate limit, but throttle anyway


def fetch_tender_detail(tender_id: str) -> dict:
    resp = _get_with_retry(f"{BASE_URL}/tenders/{tender_id}")
    return resp.json()["data"]


def is_defense_relevant(tender: dict) -> bool:
    """Client-side filter: CPV prefix OR keyword match on title/items."""
    title = (tender.get("title") or "").lower()
    items = tender.get("items", [])

    for item in items:
        cpv = item.get("classification", {}).get("id", "")
        if any(cpv.startswith(p) for p in TARGET_CPV_PREFIXES):
            return True

    haystack = title + " " + " ".join(i.get("description", "") for i in items)
    haystack = haystack.lower()
    return any(kw in haystack for kw in KEYWORDS)


def run_extraction(
    since: str | None = None, limit: int | None = None, descending: bool = False,
    overwrite: bool = False,
) -> list[str]:
    """Returns the list of tender_ids actually written to disk (new
    ones, and — when overwrite=True — updated ones too), not just a count,
    so callers like update_data.py know exactly which tenders to merge.

    `overwrite=False` (default, used for the historical backfill in
    __main__ below) skips a tender_id whose raw JSON already exists — a
    from-scratch crawl never needs to re-fetch something it already has.
    `overwrite=True` (used by update_data.py) always re-fetches, since an
    incremental sync only ever visits a tender_id because it's brand new
    or was modified since the last sync — the on-disk copy, if any, is
    exactly the stale data we're trying to refresh.
    """
    saved_ids = []
    try:
        for stub in fetch_tender_ids(since=since, descending=descending):
            tender_id = stub["id"]
            out_path = RAW_DIR / f"{tender_id}.json"
            if out_path.exists() and not overwrite:
                continue  # idempotent — skip already-fetched

            try:
                detail = fetch_tender_detail(tender_id)
            except requests.exceptions.RequestException as e:
                # Not just HTTPError — a bare ReadTimeout/ConnectionError
                # isn't an HTTPError subclass and used to crash the whole
                # run here (found live: an unattended incremental sync
                # died on one transient timeout). _get_with_retry already
                # retried once; if it still failed, skip this one tender
                # and keep going rather than lose all progress so far.
                print(f"skip {tender_id}: {e}")
                continue

            if is_defense_relevant(detail):
                out_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2))
                saved_ids.append(tender_id)
                print(f"saved {tender_id} ({len(saved_ids)})")

            time.sleep(0.1)
            if limit and len(saved_ids) >= limit:
                break
    except requests.exceptions.RequestException as e:
        # The list endpoint itself (not a single tender's detail fetch)
        # failed even after _get_with_retry's one retry — a longer outage.
        # Return what was actually saved rather than raising and losing it.
        print(f"tender list walk aborted after a persistent network error: {e}")

    return saved_ids


if __name__ == "__main__":
    # descending=True walks from today backward, so this naturally pulls
    # the most recent 5000 defense-relevant tenders rather than the oldest
    # ones — no date cutoff needed, since "most recent" is the traversal
    # order itself, not a filter on top of it.
    run_extraction(limit=5000, descending=True)

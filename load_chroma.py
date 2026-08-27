"""
load_chroma.py
Step 3 of the Prozorro Defense AI Explorer ETL pipeline.

Reads the flattened `tenders` table from prozorro.duckdb, builds one
semantically dense text chunk per tender, embeds it with a local
multilingual sentence-transformers model, and upserts it (embedding +
document text + structured metadata) into a persistent local ChromaDB
collection.

Install:
    pip install chromadb sentence-transformers duckdb pandas --break-system-packages

Usage:
    python load_chroma.py --db-path data/prozorro.duckdb --chroma-path chroma_db

Then try a query:
    python load_chroma.py --query "дрони для розвідки понад 5 млн грн"
"""

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any

import chromadb
import duckdb
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prozorro_load")

# multilingual-e5-large: best quality for Ukrainian, ~1024-dim, needs a GPU
# or patience on CPU. Swap to paraphrase-multilingual-mpnet-base-v2 (768-dim,
# ~3x lighter/faster) if you want something snappier for local iteration —
# both are free, local, and handle Ukrainian well.
DEFAULT_MODEL = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "prozorro_tenders"


# --------------------------------------------------------------------------
# Chunk construction
# --------------------------------------------------------------------------

def _fmt(row: dict, key: str, default: str = "не вказано") -> str:
    """Render a field for the text chunk, treating None/NaN/pd.NA as 'not stated'
    rather than letting Python print 'nan' or '<NA>' into the embedded text."""
    v = row.get(key)
    if v is None:
        return default
    if isinstance(v, float) and math.isnan(v):
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return str(v)


def build_chunk_text(row: dict) -> str:
    """One semantically dense chunk per tender — short enough to embed
    whole, dense enough that similarity search actually works."""
    return (
        f"Тендер: {_fmt(row, 'title')}\n"
        f"Номер: {_fmt(row, 'tender_number')}\n"
        f"Замовник: {_fmt(row, 'buyer_name')} ({_fmt(row, 'buyer_region')}), "
        f"тип замовника: {_fmt(row, 'buyer_kind')}\n"
        f"Постачальник: {_fmt(row, 'supplier_name')}\n"
        f"Категорія (CPV): {_fmt(row, 'cpv_main')} — {_fmt(row, 'cpv_description')}\n"
        f"Орієнтовний бюджет: {_fmt(row, 'budget_amount')} {_fmt(row, 'budget_currency', '')}\n"
        f"Сума контракту: {_fmt(row, 'awarded_amount')} {_fmt(row, 'awarded_currency', '')}\n"
        f"Статус: {_fmt(row, 'status')}\n"
        f"Метод закупівлі: {_fmt(row, 'procurement_method_type')}\n"
        f"Дата створення: {_fmt(row, 'date_created')}"
    )


# --------------------------------------------------------------------------
# Metadata sanitization — Chroma only accepts str/int/float/bool metadata
# values. Real duckdb output contains NaN floats, pandas <NA> (from nullable
# Int32 columns), and Timestamp objects, all of which must be normalized or
# dropped rather than passed through raw.
# --------------------------------------------------------------------------

def sanitize_metadata(row: dict) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for k, v in row.items():
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            pass  # v isn't NA-checkable (e.g. already a plain str/bool) — fine, keep going

        if isinstance(v, pd.Timestamp):
            meta[k] = v.isoformat()
        elif isinstance(v, (str, int, float, bool)):
            meta[k] = v
        else:
            meta[k] = str(v)  # last-resort stringify rather than dropping silently
    return meta


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

class Embedder:
    """Wraps a sentence-transformers model with e5-style prefix handling.

    e5 models (multilingual-e5-*) are trained with 'query: ' / 'passage: '
    prefixes and retrieve noticeably worse without them. mpnet-style models
    don't use prefixes at all — this handles both without special-casing
    call sites.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        logger.info("Loading embedding model %s (first run downloads it)...", model_name)
        self.model = SentenceTransformer(model_name)
        self.uses_e5_prefix = "e5" in model_name.lower()
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info("Model loaded, embedding dim = %d", self.dim)

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts] if self.uses_e5_prefix else texts
        vecs = self.model.encode(
            prefixed, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"query: {text}" if self.uses_e5_prefix else text
        return self.model.encode([prefixed], normalize_embeddings=True)[0].tolist()


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def load_tenders(db_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("SELECT * FROM tenders").fetchdf()
    con.close()
    logger.info("Loaded %d tenders from %s", len(df), db_path)
    return df


def load_tenders_by_ids(db_path: Path, tender_ids: list[str]) -> pd.DataFrame:
    """Used by update_data.py's incremental path — only the touched
    tender_ids need re-embedding, not the whole table."""
    if not tender_ids:
        return pd.DataFrame()
    con = duckdb.connect(str(db_path), read_only=True)
    placeholders = ",".join("?" * len(tender_ids))
    df = con.execute(f"SELECT * FROM tenders WHERE tender_id IN ({placeholders})", tender_ids).fetchdf()
    con.close()
    return df


def upsert_batch(collection, ids, embeddings, documents, metadatas):
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def embed_and_upsert(df: pd.DataFrame, collection, embedder: "Embedder", batch_size: int = 32) -> None:
    """Shared by run_load (every tender) and update_data.py (just the
    tenders that changed this run) — collection.upsert() is idempotent by
    ID either way, so re-embedding a subset never risks duplicates."""
    records = df.to_dict(orient="records")
    n = len(records)

    for start in range(0, n, batch_size):
        batch = records[start:start + batch_size]

        ids = [r["tender_id"] for r in batch]
        texts = [build_chunk_text(r) for r in batch]
        metadatas = [sanitize_metadata(r) for r in batch]
        # keep the chunk text itself queryable/inspectable via metadata too,
        # separate from Chroma's own `documents` field
        embeddings = embedder.embed_passages(texts, batch_size=batch_size)

        upsert_batch(collection, ids, embeddings, texts, metadatas)
        logger.info("Upserted %d/%d tenders", min(start + batch_size, n), n)


def run_load(db_path: Path, chroma_path: Path, model_name: str, batch_size: int):
    df = load_tenders(db_path)
    if df.empty:
        logger.error("No tenders found in %s — run the transform step first", db_path)
        sys.exit(1)

    embedder = Embedder(model_name)

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    embed_and_upsert(df, collection, embedder, batch_size)

    logger.info(
        "Done. Collection '%s' now has %d items at %s",
        COLLECTION_NAME, collection.count(), chroma_path,
    )


def run_query(chroma_path: Path, model_name: str, query: str, top_k: int = 5):
    embedder = Embedder(model_name)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(COLLECTION_NAME)

    query_vec = embedder.embed_query(query)
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)

    print(f"\nTop {top_k} results for: {query!r}\n" + "-" * 60)
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"[dist={dist:.4f}] {meta.get('title')}")
        print(f"  buyer={meta.get('buyer_name')} | amount={meta.get('awarded_amount')} {meta.get('awarded_currency', '')}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Embed and load Prozorro tenders into ChromaDB.")
    parser.add_argument("--db-path", type=Path, default=Path("data/prozorro.duckdb"))
    parser.add_argument("--chroma-path", type=Path, default=Path("chroma_db"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query", type=str, default=None, help="Skip loading, just run a similarity search")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.query:
        run_query(args.chroma_path, args.model, args.query, args.top_k)
        return

    if not args.db_path.exists():
        logger.error("DuckDB file %s does not exist — run transform_prozorro.py first", args.db_path)
        sys.exit(1)

    run_load(args.db_path, args.chroma_path, args.model, args.batch_size)


if __name__ == "__main__":
    main()

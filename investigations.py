"""
investigations.py
Pre-defined, hardcoded SQL "investigation templates" for common corruption
patterns — tender splitting, local-monopoly favoritism, and apples-to-apples
overpricing — instead of generating full SQL from scratch for these
questions.

Why templates instead of free-form text-to-SQL: rag_query.py's
generate_sql() already needed a validation gate, a retry loop, an OR-hedge
for category filtering, and several rounds of prompt fixes to become
reasonably reliable — and these specific multi-condition analytical queries
are exactly the shape a local 8B/14B model gets wrong most often. Templates
take that risk off the table for these three query types: the SQL is fixed
Python code (reviewed once here), and the LLM's only job is extracting a
parameter value (buyer name, item model) — a much smaller, much more
reliable task — which is then bound as a real SQL parameter (`?`,
never string-interpolated). That means these three query types have zero
injection surface regardless of what the LLM extracts, independent of
rag_query.py's sqlglot-based validation gate (verified live: a
`'; DROP TABLE tenders; --` value bound as a parameter is completely inert).

Also bakes in, unconditionally, two fixes that are only prompt-compliance-
dependent in the free-form generate_sql() path: the "Оборонний
постачальник" / "Приховано" masking exclusion (V5) is a literal WHERE
clause here, not something an LLM has to remember every time, and the
apples-to-apples overpricing template can only ever compare an exact
item to its own exact-item statistics, since there is no broader-category
code path for it to fall back to.

Usage:
    kind = classify_investigation(query)  # "splitting" | "monopoly" | "overpricing" | None
    if kind:
        answer = run_investigation(kind, db_path, llm_model, query)
"""

import re
from pathlib import Path

import duckdb
import pandas as pd

import llm_client

# Same masking convention as rag_query.py (V5) — baked directly into every
# template below, so (unlike free-form generated SQL) this exclusion can
# never be skipped by an LLM that didn't fully comply with a prompt rule.
_MASKED_SUPPLIER_EXCLUSION = (
    "supplier_name NOT ILIKE '%оборонний постачальник%' "
    "AND supplier_name NOT ILIKE '%приховано%'"
)

# Same floor as rag_query.py's small_sample_caveat (V9) — kept as a
# separate constant here rather than imported, to keep this module
# dependency-free of rag_query.py (see module docstring).
MIN_FAIR_SAMPLE_SIZE = 3

_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((?:https?://|www\.)[^)]*\)")
_BARE_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+")


def _strip_links(text: str) -> str:
    """Same reasoning as rag_query.py's strip_fabricated_links: no URL
    field exists anywhere in this data, so any link the LLM writes on its
    own is fabricated by construction — strip it regardless of prompt
    compliance. Applied before linkify_known_tenders() below, which then
    inserts the real, verified links."""
    text = _MD_LINK_RE.sub(r"\1", text)
    return _BARE_URL_RE.sub("", text)


# Real, verified format (same constant as rag_query.py's — duplicated
# rather than imported, see module docstring on dependency-freeness).
# Auditors need actual proof, not just aggregate stats, so every template
# below now surfaces the specific tender_number(s) involved. As with the
# vector-path fix in rag_query.py: the LLM never constructs the link
# itself (see _strip_links above) — only a tender_number that's actually
# present in the SQL result (`known_ids`, from _collect_known_tender_numbers)
# gets turned into a real clickable link. An ID-shaped token the model
# garbled or invented is left as escaped plain text, never a broken link.
PROZORRO_TENDER_URL_TEMPLATE = "https://prozorro.gov.ua/tender/{tender_number}"
_TENDER_ID_RE = re.compile(r"\bUA-\d{4}-\d{2}-\d{2}-\d{6}-[A-Za-zА-Яа-яЇїІіЄєҐґ]\b")
_MD_V1_SPECIAL_RE = re.compile(r"([_*`\[])")


def escape_markdown_v1(text: str) -> str:
    return _MD_V1_SPECIAL_RE.sub(r"\\\1", text)


def linkify_known_tenders(text: str, known_ids: set[str]) -> str:
    parts = []
    last = 0
    for m in _TENDER_ID_RE.finditer(text):
        parts.append(escape_markdown_v1(text[last:m.start()]))
        tid = m.group(0)
        if tid in known_ids:
            url = PROZORRO_TENDER_URL_TEMPLATE.format(tender_number=tid)
            parts.append(f"[{tid}]({url})")
        else:
            parts.append(escape_markdown_v1(tid))
        last = m.end()
    parts.append(escape_markdown_v1(text[last:]))
    return "".join(parts)


def _collect_known_tender_numbers(df: pd.DataFrame) -> set[str]:
    """Templates 1/2 aggregate multiple tenders per row into a
    comma-joined `tender_numbers` string (string_agg); template 3 has one
    `tender_number` per row. Handle both shapes."""
    known: set[str] = set()
    if "tender_numbers" in df.columns:
        for val in df["tender_numbers"].dropna():
            known.update(t.strip() for t in str(val).split(",") if t.strip())
    if "tender_number" in df.columns:
        known.update(str(v).strip() for v in df["tender_number"].dropna())
    return known


# --------------------------------------------------------------------------
# Router — trigger-word detection for investigation-shaped queries
# --------------------------------------------------------------------------

# Stemmed to survive Ukrainian declension (дроблення/дробленню/дробленням,
# монополія/монополії, улюблений/улюблена/улюблені), matching the style
# already used for QUANT_KEYWORDS in rag_query.py.
INVESTIGATION_TRIGGERS: dict[str, list[str]] = {
    "splitting": ["дроблен", "схем"],
    "monopoly": ["улюблен", "монопол"],
    "overpricing": ["найбільше переплатив", "переплатив за", "переплатила за", "переплата за"],
}


def classify_investigation(query: str) -> str | None:
    """Return one of {"splitting", "monopoly", "overpricing"} if the query
    matches an investigation trigger, else None. Checked before the normal
    SQL/vector router in answer_query() — these are specific, high-value
    investigation types worth a hardcoded, reliable path rather than
    free-form generation, so they should win over the generic router."""
    q = query.lower()
    for kind, markers in INVESTIGATION_TRIGGERS.items():
        if any(m in q for m in markers):
            return kind
    return None


# --------------------------------------------------------------------------
# Parameter extraction — a much smaller, more reliable task than SQL generation
# --------------------------------------------------------------------------

def extract_param(
    llm_model: str, query: str, param_label: str, param_hint: str,
    example_query: str, example_answer: str,
) -> str | None:
    """Ask the LLM to extract a single parameter value from the query.
    The result is bound as a real SQL parameter by the caller — never
    string-interpolated — so even a garbled or adversarial extraction has
    no injection surface, unlike free-form generated SQL.

    A one-shot worked example is not optional here: an earlier version of
    this prompt described the parameter abstractly ("buyer_name — замовник,
    чиїх постачальників перевіряємо") and the model answered NONE even for
    a query that named the buyer in plain text. A concrete example fixed it
    immediately — same pattern seen everywhere else in this project where
    abstract instructions alone underperform a worked example."""
    system_prompt = (
        f"Ти витягуєш параметр «{param_label}» ({param_hint}) із запитань "
        "про державні закупівлі. Виведи ЛИШЕ значення параметра — без "
        "лапок, без пояснень, одним рядком.\n\n"
        f"Приклад:\nПитання: {example_query}\nВідповідь: {example_answer}\n\n"
        "Якщо цей параметр не згаданий у питанні користувача нижче — "
        "виведи рівно слово NONE."
    )
    raw = llm_client.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        model=llm_model,
    )
    value = raw.strip().strip("\"'.")
    if not value or value.upper() == "NONE":
        return None
    return value


def _run_query(db_path: Path, sql: str, params: list) -> pd.DataFrame:
    print(f"[INVESTIGATION SQL DEBUG] params={params}\n{sql.strip()}")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(sql, params).fetchdf()
    finally:
        con.close()


# --------------------------------------------------------------------------
# Template 1: Tender splitting (дроблення)
# --------------------------------------------------------------------------
# Ukrainian procurement law exempts purchases below a threshold (~100,000
# UAH) from open competitive bidding — repeatedly awarding several
# just-under-the-threshold contracts to the same supplier is a classic way
# to dodge that requirement without ever triggering a single tender over
# the limit.

_SPLITTING_SQL_TEMPLATE = f"""
SELECT buyer_name, supplier_name, COUNT(*) AS n_contracts,
       SUM(awarded_amount) AS total_amount,
       MIN(awarded_amount) AS min_amount, MAX(awarded_amount) AS max_amount,
       string_agg(tender_number, ', ') AS tender_numbers
FROM tenders
WHERE awarded_amount BETWEEN 90000 AND 99999
  AND buyer_name IS NOT NULL AND supplier_name IS NOT NULL
  AND {_MASKED_SUPPLIER_EXCLUSION}
  {{buyer_filter}}
GROUP BY buyer_name, supplier_name
HAVING COUNT(*) >= 3
ORDER BY n_contracts DESC, total_amount DESC
LIMIT 20
"""


def run_splitting(db_path: Path, llm_model: str, query: str) -> pd.DataFrame:
    buyer = extract_param(
        llm_model, query, "buyer_name",
        "конкретний замовник, якщо згаданий — інакше NONE (перевірка йде по всій базі)",
        example_query="Чи є дроблення тендерів у Департаменту муніципальної безпеки ОМР?",
        example_answer="Департамент муніципальної безпеки ОМР",
    )
    if buyer:
        sql = _SPLITTING_SQL_TEMPLATE.format(buyer_filter="AND buyer_name ILIKE ?")
        params = [f"%{buyer}%"]
    else:
        sql = _SPLITTING_SQL_TEMPLATE.format(buyer_filter="")
        params = []
    return _run_query(db_path, sql, params)


# --------------------------------------------------------------------------
# Template 2: Local monopoly / favoritism (улюблені постачальники)
# --------------------------------------------------------------------------

_MONOPOLY_SQL = f"""
SELECT supplier_name, COUNT(*) AS n_wins, SUM(awarded_amount) AS total_amount,
       string_agg(tender_number, ', ') AS tender_numbers
FROM tenders
WHERE buyer_name ILIKE ?
  AND supplier_name IS NOT NULL
  AND {_MASKED_SUPPLIER_EXCLUSION}
GROUP BY supplier_name
ORDER BY n_wins DESC, total_amount DESC
LIMIT 20
"""


def run_monopoly(db_path: Path, llm_model: str, query: str) -> pd.DataFrame | None:
    buyer = extract_param(
        llm_model, query, "buyer_name",
        "замовник, чиїх постачальників перевіряємо на фаворитизм",
        example_query="Чи є монополія постачальників для Департаменту муніципальної безпеки ОМР?",
        example_answer="Департамент муніципальної безпеки ОМР",
    )
    if not buyer:
        return None
    return _run_query(db_path, _MONOPOLY_SQL, [f"%{buyer}%"])


# --------------------------------------------------------------------------
# Template 3: Apples-to-apples overpricing for one exact item
# --------------------------------------------------------------------------
# Unlike generate_sql()'s free-form CTE pattern (which an LLM could still
# widen to a whole category — see V9's fix), this template structurally
# can only ever compare an item to statistics computed over that exact
# same ILIKE pattern — there is no broader-category code path here at all.

_OVERPRICING_SQL = """
WITH item_stats AS (
    SELECT AVG(unit_amount) AS avg_unit_amount,
           MEDIAN(unit_amount) AS median_unit_amount,
           MAX(unit_amount) AS max_unit_amount,
           COUNT(*) AS sample_size
    FROM items
    WHERE description ILIKE ? AND unit_amount IS NOT NULL
)
SELECT t.tender_number, i.description, i.unit_amount,
       s.avg_unit_amount, s.median_unit_amount, s.sample_size,
       (i.unit_amount / s.median_unit_amount) AS ratio_to_median
FROM items i
JOIN item_stats s ON TRUE
JOIN tenders t ON t.tender_id = i.tender_id
WHERE i.description ILIKE ?
  AND i.unit_amount = s.max_unit_amount
"""


def run_overpricing(db_path: Path, llm_model: str, query: str) -> pd.DataFrame | None:
    item = extract_param(
        llm_model, query, "item_name",
        "конкретна модель чи тип товару, напр. 'Mavic 3T'",
        example_query="Чи є переплата за Mavic 3T?",
        example_answer="Mavic 3T",
    )
    if not item:
        return None
    pattern = f"%{item}%"
    return _run_query(db_path, _OVERPRICING_SQL, [pattern, pattern])


# --------------------------------------------------------------------------
# Narration
# --------------------------------------------------------------------------

_EMPTY_RESULT_TEXT = {
    "splitting": (
        "Ознак дроблення тендерів (3+ контракти по 90 000-99 999 грн від "
        "одного замовника одному постачальнику) у базі не знайдено."
    ),
    "monopoly": (
        "Не вдалося знайти дані по цьому замовнику, або в нього немає "
        "повторюваних постачальників у базі."
    ),
    "overpricing": "Недостатньо даних по цій моделі товару для порівняння цін.",
}

_MISSING_PARAM_TEXT = {
    "monopoly": (
        "Для перевірки монополії постачальників потрібно вказати "
        "конкретного замовника. Наприклад: «Чи є монополія постачальників "
        "для Департаменту муніципальної безпеки ОМР?»"
    ),
    "overpricing": (
        "Для перевірки переплати потрібно вказати конкретну модель чи тип "
        "товару. Наприклад: «Чи є переплата за Mavic 3T?»"
    ),
}

# A "⚠️" finding without the specific tenders it's based on isn't
# evidence an auditor can act on — every focus below now requires citing
# the exact tender_number(s) from the table (they're real column values
# here, not something to construct: string_agg in templates 1/2, a plain
# column in template 3). Write them as plain text, NOT as markdown links
# — narrate_investigation() turns each one that's actually in the table
# into a real clickable link afterward; see linkify_known_tenders above.
_TENDER_NUMBER_INSTRUCTION = (
    "ОБОВ'ЯЗКОВО процитуй КОНКРЕТНІ номери тендерів (стовпець "
    "tender_number або tender_numbers) як доказ — аудитору потрібні "
    "номери тендерів, а не лише узагальнена статистика. Пиши номер "
    "тендера як є, звичайним текстом (напр. UA-2023-12-20-016647-a), БЕЗ "
    "markdown-посилань — система сама перетворить його на клікабельне "
    "посилання."
)

_INVESTIGATION_FOCUS = {
    "splitting": (
        "Це результат перевірки на ДРОБЛЕННЯ ТЕНДЕРІВ — кілька контрактів "
        "по 90 000-99 999 грн (щоб уникнути порогу відкритих торгів) від "
        "одного замовника одному постачальнику. Назви конкретного "
        "замовника й постачальника, кількість контрактів і загальну суму. "
        "Використовуй фразу «⚠️ Можливе дроблення тендеру» лише якщо дані "
        "дійсно це підтверджують (3+ контракти в цьому вузькому діапазоні). "
        f"{_TENDER_NUMBER_INSTRUCTION}"
    ),
    "monopoly": (
        "Це результат перевірки на ЛОКАЛЬНУ МОНОПОЛІЮ — найчастіших "
        "постачальників для конкретного замовника. Якщо один постачальник "
        "явно домінує за кількістю перемог чи сумою порівняно з іншими — "
        "це ознака фаворитизму. Використовуй фразу «⚠️ Можлива аномалія / "
        f"Ризик переплати» лише якщо домінування дійсно виражене. {_TENDER_NUMBER_INSTRUCTION}"
    ),
    "overpricing": (
        "Це результат перевірки на ПЕРЕПЛАТУ — найдорожчий екземпляр "
        "конкретної моделі товару порівняно з медіаною/середнім по ТІЙ "
        "САМІЙ моделі (apples-to-apples, а не по всій категорії). "
        "Використовуй фразу «⚠️ Можлива аномалія / Ризик переплати» лише "
        f"якщо ratio_to_median дійсно суттєво вищий за 1. {_TENDER_NUMBER_INSTRUCTION}"
    ),
}


def narrate_investigation(llm_model: str, kind: str, query: str, df: pd.DataFrame) -> str:
    if df.empty:
        return _EMPTY_RESULT_TEXT[kind]

    table_text = df.to_string(index=False)
    system_prompt = (
        "Ти — гострий журналіст-розслідувач з питань антикорупції у сфері "
        "державних закупівель України. Наведи відповідь на основі ЛИШЕ "
        "даних таблиці нижче — реального результату SQL-запиту до бази "
        "тендерів Prozorro. Не вигадуй чисел чи назв компаній, яких немає "
        "в таблиці. Відповідай українською, гранично стисло, буліт-пойнти. "
        "НІКОЛИ не починай канцелярсько («На підставі наданого "
        "контексту...» тощо) — одразу факти.\n\n"
        f"{_INVESTIGATION_FOCUS[kind]}\n\n"
        "Якщо в таблиці є supplier_name = «Оборонний постачальник» чи "
        "«Приховано» — це офіційне маскування Prozorro для оборонних "
        "закупівель, засекречених з міркувань нацбезпеки, а НЕ реальна "
        "компанія; поясни це, а не роби висновок про монополію одного "
        "постачальника."
    )
    prompt = f"Питання: {query}\n\nРезультат SQL-запиту:\n{table_text}\n\nВідповідь:"
    raw = llm_client.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        model=llm_model,
    )
    answer = _strip_links(raw)
    known_ids = _collect_known_tender_numbers(df)
    cited_ids = {tid for tid in known_ids if tid in answer}
    answer = linkify_known_tenders(answer, known_ids)

    # Deterministic backstop: an auditor asked for this feature specifically
    # because aggregate stats without tender links aren't proof — observed
    # live that the LLM sometimes omits citing them even when told to
    # (rule compliance, not a data problem: the SQL always has them). Rather
    # than let the actual evidence depend on whether the model felt like
    # mentioning it, guarantee it's always appended when the narration
    # didn't already cite at least one of the real tender numbers.
    if known_ids and not cited_ids:
        proof_links = ", ".join(
            f"[{tid}]({PROZORRO_TENDER_URL_TEMPLATE.format(tender_number=tid)})"
            for tid in sorted(known_ids)
        )
        answer = f"{answer}\n\n📎 Тендери: {proof_links}"

    if "sample_size" in df.columns:
        try:
            min_n = pd.to_numeric(df["sample_size"], errors="coerce").min()
        except Exception:
            min_n = None
        if min_n is not None and pd.notna(min_n) and min_n < MIN_FAIR_SAMPLE_SIZE:
            answer = (
                f"ℹ️ Розмір вибірки замалий (N={int(min_n)}) — висновок "
                f"про аномалію може бути ненадійним.\n\n{answer}"
            )

    return answer


def run_investigation(kind: str, db_path: Path, llm_model: str, query: str) -> str:
    if kind == "splitting":
        df = run_splitting(db_path, llm_model, query)
    elif kind == "monopoly":
        df = run_monopoly(db_path, llm_model, query)
        if df is None:
            return _MISSING_PARAM_TEXT["monopoly"]
    elif kind == "overpricing":
        df = run_overpricing(db_path, llm_model, query)
        if df is None:
            return _MISSING_PARAM_TEXT["overpricing"]
    else:
        raise ValueError(f"Unknown investigation kind: {kind!r}")

    return narrate_investigation(llm_model, kind, query, df)

"""
rag_query.py
Step 4 of the Prozorro Defense AI Explorer pipeline — hybrid RAG.

A keyword router splits incoming questions into two paths:
  - quantitative (sums/averages/counts/rankings) -> text-to-SQL against the
    DuckDB `tenders`/`items` tables, validated read-only by sqlglot, then
    narrated by the LLM.
  - qualitative (everything else) -> vector search over the ChromaDB
    collection built in Step 3, answered by the LLM from retrieved context.

Both paths refuse to guess: the vector path won't answer from irrelevant
context (see RELEVANCE_THRESHOLD), and the SQL path won't execute anything
that isn't a single validated read-only SELECT.

Install:
    pip install ollama sqlglot
    (chromadb / sentence-transformers / duckdb already installed from
    earlier steps)

Ollama setup (one-time):
    # macOS / Linux
    curl -fsSL https://ollama.com/install.sh | sh
    # Windows: download the installer from https://ollama.com/download

    ollama pull qwen2.5:14b-instruct   # or a lighter model, see README notes below
    ollama list                        # confirm it's there

Usage:
    python rag_query.py                              # interactive mode
    python rag_query.py --query "середня вартість дронів у 2024 році"
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import chromadb
import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp

import llm_client

# Reuse the exact same embedding logic used at load time — retrieval quality
# depends on query and passage embeddings coming from the identical model
# and prefix convention (see Embedder in load_chroma.py).
from load_chroma import Embedder, DEFAULT_MODEL, COLLECTION_NAME
from investigations import classify_investigation, run_investigation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prozorro_rag")

DEFAULT_LLM_MODEL = "qwen2.5:14b-instruct"
DEFAULT_DB_PATH = Path("data/prozorro.duckdb")

# Results above this cosine distance are treated as "not actually relevant"
# rather than forced into the context.
#
# Re-calibrated against the 1047-tender dataset (multilingual-e5-large) by
# probing 10 known-present categories (drones, generators, medical
# materials, uniforms, fire/rescue equipment, etc.) against 8 known-absent
# ones (submarines, hosting services, wedding banquets, ...) and comparing
# top-k distance distributions. Unlike the 43-tender sample, there is NO
# longer a clean separating gap: genuine same-category hits range up to
# ~0.198 within the default top_k=8 (e.g. a real "DJI Mavic 3 Enterprise"
# drone tender at 0.1974 for the query "дрони для розвідки"), while a
# handful of absent-category queries land as low as ~0.177-0.182 purely by
# incidental text overlap (e.g. "закупівля космічного телескопа" nearest-
# matches an aircraft CPV description that happens to mention "космічні
# апарати"). A single scalar threshold cannot separate these two groups
# without sacrificing one of them.
#
# 0.20 is chosen to prioritize recall — do not silently drop genuine hits
# in populated categories, which was the actual failure being fixed here —
# accepting that a small number of semantically-adjacent misses will now
# clear this gate. The SYSTEM_PROMPT's "insufficient context, don't
# fabricate" instruction is the second-line defense against those residual
# false positives; verify it's still holding (e.g. with an absent-category
# query like "послуги хостингу веб-сайтів") whenever this constant changes.
#
# STALE ASSUMPTIONS (as of the top_k=8->5 latency change): this calibration
# was done at top_k=8 against a 1047-tender dataset; the dataset has since
# grown to 6047 (6x, unevenly distributed across time) and top_k dropped to
# 5. Both change what "within top_k" means for recall — re-run the same
# probe-known-categories method before trusting this number again.
RELEVANCE_THRESHOLD = 0.20

SYSTEM_PROMPT = (
    "Ти — не пошуковий асистент, а гострий журналіст-розслідувач з питань "
    "антикорупції у сфері державних закупівель України. Відповідай ЛИШЕ на "
    "основі наведеного нижче контексту з реальних тендерів Prozorro. Якщо в "
    "контексті недостатньо даних для точної відповіді — прямо скажи про це, "
    "не вигадуй цифри чи факти. Завжди вказуй номери тендерів (поле "
    "'Номер'), на які спираєшся у відповіді.\n\n"
    "КРИТИЧНО ВАЖЛИВО (точність):\n"
    "1. Власні назви (замовник, постачальник, регіон) переписуй ІЗ КОНТЕКСТУ "
    "ДОСЛІВНО, символ у символ. НІКОЛИ не 'розшифровуй' і не доповнюй "
    "скорочення своїми здогадками (наприклад, абревіатура «ОМР» у назві "
    "замовника — це частина офіційної назви української установи, а не "
    "натяк на якесь інше місто чи країну; якщо не певен, що означає "
    "скорочення, залиш його як є, не інтерпретуючи).\n"
    "2. НІКОЛИ не пиши URL чи markdown-посилання [текст](url) сам — "
    "система автоматично перетворює номер тендера (напр. "
    "UA-2023-12-20-016647-a) на клікабельне посилання, якщо ти просто "
    "напишеш номер як звичайний текст. Будь-яке посилання, написане тобою "
    "вручну, буде вигаданим — просто цитуй номер тендера, більше нічого "
    "не потрібно.\n\n"
    "РОЛЬ АНТИКОРУПЦІЙНОГО РОЗСЛІДУВАЧА (обов'язково для питань про "
    "конкретний товар чи постачальника):\n"
    "3. Порівнюй суми/ціни між усіма наведеними в контексті тендерами на "
    "той самий товар. Якщо в якогось одного тендера ціна за одиницю чи "
    "бюджет ПОМІТНО вищі за інші (лише коли це видно з реальних чисел у "
    "контексті, не вигадуй різницю) — це потенційна переплата.\n"
    "4. Якщо один і той самий постачальник трапляється в кількох наведених "
    "тендерах — познач це як можливу ознаку слабкої конкуренції.\n"
    "   ВИНЯТОК: «Оборонний постачальник» (або «Приховано») у полі "
    "Постачальник — це НЕ назва реальної компанії, а офіційне маскування "
    "Prozorro для оборонних закупівель, засекречених з міркувань "
    "нацбезпеки; під ним ховаються десятки різних реальних постачальників. "
    "НІКОЛИ не позначай це як монополію чи домінування — якщо бачиш цей "
    "запис, поясни, що дані замасковані, а не роби висновок про одну "
    "компанію.\n"
    "5. Коли знаходиш таку аномалію — використовуй У ВІДПОВІДІ ТОЧНУ фразу "
    "«⚠️ Можлива аномалія / Ризик переплати» поруч із конкретним фактом. "
    "Якщо аномалій у наведених даних немає — просто не вживай цю фразу, не "
    "видумуй привід її вжити.\n"
    "6. Формат: гранично стисло, буліт-пойнти. НІКОЛИ не починай канцелярсько "
    "— заборонені фрази на кшталт «На підставі наданого контексту...», "
    "«Based on the provided context...» тощо. Одразу факти.\n"
    "7. Контекст містить лише короткий чанк на тендер (назва, замовник, "
    "категорія, бюджет, статус) — НЕ повні технічні специфікації товару. "
    "Якщо користувач просить глибокі технічні характеристики/специфікації "
    "(напр. точну вагу, дальність польоту, матеріал, стандарти) — "
    "ОБОВ'ЯЗКОВИЙ ПОРЯДОК дій, без винятків:\n"
    "   (1) СПОЧАТКУ дай ЗМІСТОВНУ відповідь на основі того, що ФАКТИЧНО "
    "є в контексті — перелічи знайдені тендери, їхні назви, категорії, "
    "бюджети, статуси, будь-які деталі з чанків, що стосуються питання. "
    "Це ОБОВ'ЯЗКОВА частина відповіді, навіть якщо це не є 'глибокими "
    "специфікаціями', яких просив користувач.\n"
    "   (2) ТІЛЬКИ ПІСЛЯ ЦЬОГО, в самому кінці, окремим реченням, додай "
    "застереження: «Зауважте: база містить лише короткі описи тендерів. "
    "Повні технічні специфікації знаходяться у прикріплених "
    "PDF-документах на сайті Prozorro, які наразі не аналізуються.»\n"
    "   Застереження НІКОЛИ не замінює відповідь із кроку (1) — відповідь "
    "лише з застереження і без жодної спроби використати наявний "
    "контекст вважається ПОМИЛКОЮ.\n"
    "   НАЙВАЖЛИВІШЕ ОБМЕЖЕННЯ (пріоритетніше за 'дай змістовну "
    "відповідь' вище): у кроці (1) використовуй ЛИШЕ цифри й факти, що "
    "буквально написані в наведеному контексті. Якщо контекст називає "
    "модель товару (напр. 'DJI Mavic 3') без ваги/дальності польоту — "
    "НІКОЛИ не підставляй ці числа зі своїх загальних знань про бренд чи "
    "модель, навіть якщо ти 'знаєш' типові характеристики цього товару з "
    "власного навчання. Конкретна закупівля може відрізнятися "
    "комплектацією чи версією від загальновідомої моделі, а вигадана "
    "цифра, яка звучить правдоподібно, тут НЕБЕЗПЕЧНІША за чесне "
    "визнання прогалини. Якщо конкретної цифри немає в контексті — прямо "
    "напиши, що контекст її не містить, замість підстановки з пам'яті.\n"
    "8. Формат виводу — як у сучасному, зручному Telegram-боті:\n"
    "   - Використовуй емодзі для структури: 🚁/📦 перед товаром чи "
    "назвою тендера, 🏢 перед замовником, 💰 перед сумою/бюджетом, 🏆 "
    "перед постачальником.\n"
    "   - Гроші пиши без зайвих нулів і крапок: не '3047499.99 UAH', а "
    "'3 047 500 грн' (округлено, пробіли між розрядами); для великих сум "
    "використовуй скорочення — не '46700000 грн', а '46.7 млн грн'.\n"
    "   - Максимально стисло — довгий текст читають гірше, і кожне зайве "
    "речення — це секунди очікування для користувача. Не повторюй одне й "
    "те саме різними словами.\n"
    "9. Коли питання стосується технічних характеристик чи вимог тендера — "
    "ПРОАКТИВНО аналізуй текст на предмет ДИСКРИМІНАЦІЙНИХ УМОВ: "
    "надмірно вузькі/специфічні розміри, вага, унікальні особливості чи "
    "точні технічні параметри, які виглядають підігнаними під ОДНОГО "
    "конкретного виробника чи модель (а не під клас товару загалом) — "
    "класична схема обмеження конкуренції. Оцінюй лише те, що реально "
    "видно з тексту контексту, не вигадуй деталей. Якщо бачиш ознаки "
    "цього — познач ЯВНО фразою «⚠️ Можливі дискримінаційні вимоги / "
    "Заточка під одного виробника» разом із конкретним прикладом умови з "
    "контексту. Якщо ознак немає — не вживай цю фразу."
)


# --------------------------------------------------------------------------
# Router — keyword-based classification of quantitative vs qualitative intent
# --------------------------------------------------------------------------

# Deliberately narrow to aggregation/ranking/comparison signals rather than
# generic words like "скільки" alone, which shows up in both "скільки коштує
# X" (a lookup, answerable from a single retrieved chunk) and "скільки
# тендерів У 2024" (a count, needs SQL). This is a heuristic, not a
# classifier — tune the list as real queries surface false routes.
QUANT_KEYWORDS = [
    "скільки тендерів", "скільки закупівель", "скільки контрактів",
    "кількість", "сумарн", "загальна сума", "загальний бюджет",
    "середн", "медіан", "максимальн", "мінімальн",
    "найдорожч", "найдешевш", "найчастіше", "топ ", "топ-", "рейтинг",
    "порівня", "відсот", "розподіл", "згрупу", "статистик",
    "хто", "постачальник", "конкуренція",
    "count(", "sum(", "avg(", "average", "median", "total", "how many",
]


# These signal a request for descriptive/qualitative content (specs,
# requirements, free-text description) and should win over a QUANT_KEYWORDS
# match — e.g. "які ... найчастіше ставлять вимоги" contains "найчастіше"
# (a quant signal) but is asking for text content, not a number.
QUALITATIVE_OVERRIDE_KEYWORDS = ["характеристики", "вимоги", "опис", "технічні"]

# "який"/"які" are checked separately and ONLY as the query's leading word.
# Mid-sentence they're just as likely to be a relative pronoun inside an
# otherwise-quantitative question — e.g. "...замовника, який витратив
# найбільшу суму" is asking for MAX(), not a qualitative description.
# Treating them as an anywhere-substring override (like the words above)
# would silently misroute that kind of query back to vector search.
QUALITATIVE_LEADING_WORDS = ("який ", "які ")


def classify_query(query: str) -> str:
    """Return 'sql' for quantitative/aggregate questions, 'vector' otherwise."""
    q = query.strip().lower()
    if q.startswith(QUALITATIVE_LEADING_WORDS):
        return "vector"
    if any(kw in q for kw in QUALITATIVE_OVERRIDE_KEYWORDS):
        return "vector"
    if any(kw in q for kw in QUANT_KEYWORDS):
        return "sql"
    return "vector"


# The bot is stateless — no conversation history is kept between messages
# (see answer_query). A query that only makes sense as a reply to a prior
# answer (a bare pronoun/reference, "why?", "the same one?") would
# otherwise get routed into classify_query and generate a SQL/vector
# query with no idea what "this" or "the same" refers to — a guaranteed
# broken or nonsensical result. These markers catch phrasing that's
# reference-shaped rather than a self-contained question. Deliberately
# narrow (e.g. "чому це"/"чому так", not bare "чому") so a genuine new
# analytical question like "Чому середня ціна генераторів вища?" — which
# names its own subject — doesn't get misclassified as a follow-up.
FOLLOWUP_MARKERS = [
    "чому це", "чому так", "чому?", "а це", "а той", "а всі",
    "той самий", "та сама", "ті самі", "це те саме",
    "саме цей", "саме ця", "саме ці", "саме той",
    "справді?", "точно?", "серйозно?",
    "is that the same", "is this the same", "same one", "same item",
    "why is that", "why is this", "why?",
]

FOLLOWUP_FALLBACK_TEXT = (
    "⚠️ Я поки що не зберігаю історію діалогу. Будь ласка, сформулюйте "
    "ваше питання як новий, повний запит (наприклад: «Яка середня ціна "
    "саме на квадрокоптери Mavic 3?»)."
)


def is_conversational_followup(query: str) -> bool:
    q = query.strip().lower()
    return any(marker in q for marker in FOLLOWUP_MARKERS)


# --------------------------------------------------------------------------
# Text-to-SQL generation
# --------------------------------------------------------------------------

ALLOWED_TABLES = {"tenders", "items"}

# Functions DuckDB could use to read arbitrary files or attach other
# databases. The exp.Select/exp.Union type check below already rejects
# ATTACH/COPY/PRAGMA/INSERT/etc. (they parse to different expression types),
# but table-valued functions can appear *inside* a FROM clause of an
# otherwise-valid SELECT, so they need an explicit second check.
DANGEROUS_FUNCS = {
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
    "glob", "sqlite_scan", "postgres_scan", "attach", "system", "install", "load",
}


def get_schema_description(db_path: Path) -> str:
    """Pull live column names/types from DuckDB so the prompt never drifts
    out of sync with the actual schema."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        lines = []
        for table in ("tenders", "items"):
            cols = con.execute(f"DESCRIBE {table}").fetchdf()
            col_list = ", ".join(f"{r.column_name} ({r.column_type})" for r in cols.itertuples())
            lines.append(f"Таблиця `{table}`: {col_list}")
        return "\n".join(lines)
    finally:
        con.close()


def extract_sql(text: str) -> str:
    """Pull the SQL statement out of raw LLM output, stripping markdown code
    fences and any leading chatter the model adds despite instructions not to."""
    text = text.strip()
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    upper = text.upper()
    # A CTE query starts with WITH, not SELECT — truncating at the first
    # SELECT (as this used to do unconditionally) chops off a leading
    # `WITH x AS (...)` clause entirely, turning valid CTE SQL into a
    # syntax error. Cut at whichever legitimate start keyword appears first.
    starts = [i for i in (upper.find("WITH "), upper.find("SELECT")) if i != -1]
    if starts:
        start = min(starts)
        if start > 0:
            text = text[start:]
    return text.strip().rstrip(";").strip()


def generate_sql(llm_model: str, query: str, schema_desc: str, messages: list[dict] | None = None) -> tuple[str, list[dict]]:
    """Generate SQL for `query`. On the first call (messages=None), builds a
    fresh system+user prompt. On a retry, pass back the conversation list
    returned by the previous call with a trailing user turn describing what
    went wrong (see answer_sql_query) — the model then sees its own failed
    SQL and the actual error, and can self-correct instead of repeating the
    same mistake blind. Returns (extracted_sql, updated_messages) so the
    caller can keep chaining retries."""
    if messages is not None:
        raw = llm_client.chat(messages=messages, model=llm_model)
        messages = messages + [{"role": "assistant", "content": raw}]
        return extract_sql(raw), messages

    system_prompt = (
        "Ти генератор SQL для DuckDB. Тобі доступні ЛИШЕ ці таблиці:\n"
        f"{schema_desc}\n\n"
        "Таблиці зв'язані через tender_id. ВАЖЛИВО: усі текстові дані в базі "
        "(title, description, cpv_description, buyer_name, supplier_name тощо) "
        "записані УКРАЇНСЬКОЮ мовою — це дані реальних тендерів Prozorro, "
        "англійського тексту в них немає.\n\n"
        "КРИТИЧНО ВАЖЛИВО (не копіюй приклади): нижче наведені приклади SQL "
        "(з такими товарами, як «генератори» чи «офісний папір») ілюструють "
        "ЛИШЕ СИНТАКСИС і СТРУКТУРУ запиту — конкретний товар у них "
        "випадковий і НЕ має жодного стосунку до реального питання "
        "користувача. Ти ЗОБОВ'ЯЗАНИЙ самостійно визначити РЕАЛЬНИЙ товар "
        "чи категорію з фактичного питання нижче (напр. дрони, медикаменти, "
        "транспорт, папір — що завгодно) і підставити САМЕ ЙОГО у "
        "WHERE/ILIKE-умову. НІКОЛИ не копіюй товар із прикладу, якщо "
        "користувач питає про щось інше — це найпоширеніша й найкритичніша "
        "помилка, якої треба уникати.\n\n"
        "Правила:\n"
        "1. Згенеруй РІВНО ОДИН read-only SELECT-запит, що відповідає на питання.\n"
        "2. Використовуй ЛИШЕ таблиці tenders та items — жодних інших джерел даних.\n"
        "3. НІКОЛИ не використовуй англійські слова у LIKE чи = умовах для "
        "текстових колонок. Якщо питання сформульоване англійською або "
        "потребує пошуку за поняттям — спочатку подумки перекладай поняття "
        "українською і шукай саме українськими словами/синонімами "
        "(наприклад, замість LIKE '%medical equipment%' пиши "
        "LIKE '%медичне обладнання%').\n"
        "4. КРИТИЧНО: для фільтрації за категорією товарів/послуг ЗАВЖДИ "
        "використовуй cpv_main-префікс У ПОЄДНАННІ З текстовим пошуком "
        "через OR. НІКОЛИ не покладайся ЛИШЕ на текстовий пошук "
        "(cpv_description/title/description) без cpv_main — текстовий "
        "збіг завжди вужчий за реальний обсяг категорії (наприклад, "
        "лише за словом у назві можна пропустити десятки тендерів, чий "
        "заголовок сформульований інакше, але які належать до тієї ж "
        "CPV-категорії), тому запит без cpv_main систематично занижує "
        "результат — це так само помилково, як і вигаданий CPV-код.\n"
        "   Довідник перевірених префіксів cpv_main у цій базі (використовуй "
        "як основний орієнтир, а не вигадуй інші коди для цих категорій):\n"
        "     33 — медичне обладнання, медикаменти, фармацевтика\n"
        "     18 — одяг, обмундирування, спецодяг\n"
        "     35 — охорона, протипожежне/рятувальне обладнання, "
        "радіоелектронний захист\n"
        "     34 — транспорт, дрони/безпілотні літальні апарати, "
        "вертольоти/літаки\n"
        "     31 — генератори, електрообладнання, акумулятори\n"
        "     30 — офісне приладдя, папір\n"
        "   Якщо категорія користувача НЕ входить у цей список — приблизно "
        "вгадай найближчий розділ, але це ТИМ БІЛЬШЕ причина обов'язково "
        "додати текстовий пошук через OR, а не покладатися лише на здогад.\n"
        "   ВАЖЛИВО про колонки для текстової частини: cpv_description — "
        "це ОФІЦІЙНА назва CPV-розділу (напр. «Безпілотні літальні "
        "апарати»), а НЕ вільний опис товару — вона рідко містить "
        "розмовні слова («дрон», «квадрокоптер» тощо). title — це "
        "власна назва тендера, яку писав замовник, і САМЕ ТАМ "
        "найімовірніше зустрінеться розмовне слово з питання "
        "користувача. Тому комбінуй ВСІ ТРИ умови через OR:\n"
        "   Приклад (СИНТАКСИС, не тема запиту): категорія 'генератори' "
        "відповідає розділу CPV 31 з довідника вище, тож пиши:\n"
        "     cpv_main LIKE '31%' OR cpv_description ILIKE '%генератор%' "
        "OR title ILIKE '%генератор%'\n"
        "   а НЕ лише `title ILIKE '%генератор%'` чи "
        "`cpv_description ILIKE '%генератор%'` окремо — без cpv_main "
        "запит поверне лише частину реальних даних.\n"
        "   а НЕ лише `cpv_main LIKE '31%'` саме по собі.\n"
        "5. tenders і items пов'язані один-до-багатьох (один тендер може мати "
        "кілька рядків у items). НЕ приєднуй items, якщо питання не потребує "
        "колонок items (quantity, unit_amount, line_total, delivery_region "
        "тощо) — інакше SUM/AVG/COUNT над колонками tenders (awarded_amount, "
        "budget_amount) порахує один і той самий тендер кілька разів, по "
        "одному на кожен рядок items. Для агрегатів на рівні тендера рахуй "
        "ЛИШЕ по таблиці tenders без JOIN. Приєднуй items тільки тоді, коли "
        "потрібні саме дані про окремі позиції закупівлі.\n"
        "   НЕПРАВИЛЬНО (завищує суму через дублювання по items; товар "
        "у прикладі — офісний папір, це лише СИНТАКСИС):\n"
        "     SELECT SUM(t.awarded_amount) FROM tenders t "
        "JOIN items i ON t.tender_id = i.tender_id WHERE t.cpv_main LIKE '30%'\n"
        "   ПРАВИЛЬНО (кожен тендер рахується рівно один раз):\n"
        "     SELECT SUM(awarded_amount) FROM tenders WHERE cpv_main LIKE '30%'\n"
        "   Якщо УМОВА фільтрації стосується колонки items (наприклад, "
        "items.cpv точніший за tenders.cpv_main), а АГРЕГАТ все одно "
        "рахується по колонках tenders — це теж НЕ привід для JOIN. "
        "Використовуй підзапит IN або EXISTS замість JOIN:\n"
        "   НЕПРАВИЛЬНО (той самий фан-аут, лише умова тепер з items):\n"
        "     SELECT AVG(t.awarded_amount) FROM tenders t "
        "JOIN items i ON t.tender_id = i.tender_id "
        "WHERE t.cpv_main LIKE '30%' OR i.cpv LIKE '30%'\n"
        "   ПРАВИЛЬНО (items лише фільтрує, кожен тендер рахується один раз):\n"
        "     SELECT AVG(awarded_amount) FROM tenders WHERE cpv_main LIKE '30%' "
        "OR tender_id IN (SELECT tender_id FROM items WHERE cpv LIKE '30%')\n"
        "6. Якщо запит повертає окремі рядки (не агрегат) — додай LIMIT 20.\n"
        "7. Якщо потрібно порівняти окремі рядки із середнім/медіанним "
        "значенням (наприклад, «ціна суттєво вища за середню») — "
        "використовуй CTE через ключове слово WITH, а потім приєднуй його "
        "через JOIN ... ON true (CTE з агрегатом повертає рівно один "
        "рядок, тому звичайна умова з'єднання не потрібна). НЕ ЗАБУВАЙ "
        "саме слово WITH перед визначенням CTE — без нього DuckDB поверне "
        "помилку синтаксису.\n"
        "   КРИТИЧНО (порівнюй яблука з яблуками, не з апельсинами): "
        "НІКОЛИ не рахуй середнє по ШИРОКІЙ категорії (напр. усі items з "
        "cpv LIKE '34%') і не порівнюй з нею КОНКРЕТНИЙ товар — широка "
        "категорія змішує дешеві дрібниці (пропелери, акумулятори "
        "окремо) з дорогими комплексними системами (напр. комплект із "
        "кількох дронів + наземна станція керування), тому середнє "
        "виходить штучно заниженим і будь-який складний комплект "
        "виглядає як 'аномальна переплата', хоча насправді це просто "
        "інший, значно дорожчий тип товару. Замість цього звужуй CTE "
        "середнього до СХОЖИХ товарів — використовуй конкретні ключові "
        "слова з опису самого товару, що перевіряється (напр. якщо "
        "перевіряєш 'Mavic 3', рахуй середнє ЛИШЕ по "
        "description ILIKE '%Mavic 3%', а не по всій категорії дронів).\n"
        "   Також ЗАВЖДИ додавай до CTE середнього колонку "
        "sample_size — COUNT(*) рядків, що увійшли в середнє. Якщо "
        "sample_size замалий (менше 3) для чесного порівняння — це "
        "означає, що даних недостатньо для висновку про аномалію; "
        "прямо напиши це користувачу замість того, щоб позначати "
        "цінову різницю як переплату.\n"
        "   ПРАВИЛЬНИЙ приклад СИНТАКСИСУ (товар тут — офісний папір, "
        "довільний; підставляй РЕАЛЬНИЙ товар з питання користувача, "
        "звужений до конкретної моделі/типу, а не до всієї категорії):\n"
        "     WITH avg_price AS (\n"
        "       SELECT AVG(unit_amount) AS avg_unit_amount, "
        "COUNT(*) AS sample_size FROM items\n"
        "       WHERE description ILIKE '%офісний папір А4%'\n"
        "     )\n"
        "     SELECT i.tender_id, i.description, i.unit_amount, "
        "ap.avg_unit_amount, ap.sample_size\n"
        "     FROM items i\n"
        "     JOIN avg_price ap ON true\n"
        "     WHERE i.description ILIKE '%офісний папір А4%'\n"
        "       AND i.unit_amount > ap.avg_unit_amount * 1.3\n"
        "     ORDER BY i.unit_amount DESC\n"
        "     LIMIT 20\n"
        "   (Ще раз: «офісний папір А4» тут випадковий приклад — товар і "
        "рівень деталізації підставляй з реального питання користувача. "
        "Якщо користувач питає про дрони, звужуй до конкретної моделі чи "
        "типу дрона, згаданого у нього ('Mavic 3', 'FPV одноразового "
        "типу' тощо), а НЕ до всієї категорії 'дрони'/'34%'. НІКОЛИ не "
        "залишай товар з прикладу.)\n"
        "8. Коли групуєш (GROUP BY) чи агрегуєш за текстовим полем, яке може "
        "бути порожнім (supplier_name, buyer_name тощо) — ЗАВЖДИ додавай "
        "умову `<поле> IS NOT NULL` у WHERE. Без цього рядки з відсутнім "
        "постачальником/замовником групуються в один безглуздий рядок "
        "'None'/NULL, який засмічує результат і може навіть очолити "
        "рейтинг за кількістю.\n"
        "   Додатково: значення supplier_name = 'Оборонний постачальник' "
        "(також трапляється 'Приховано') — це НЕ реальна компанія, а "
        "офіційне маскування Prozorro для оборонних закупівель, засекречених "
        "з міркувань національної безпеки. Під цим написом ховаються десятки "
        "РІЗНИХ реальних постачальників одночасно. Коли запит стосується "
        "концентрації/монополії постачальників (хто виграє найчастіше тощо) "
        "— ЗАВЖДИ додатково виключай і це значення, інакше воно штучно "
        "виглядає як один домінантний постачальник, хоча насправді це "
        "об'єднання багатьох замаскованих:\n"
        "     SELECT supplier_name, COUNT(*) AS wins FROM tenders\n"
        "     WHERE supplier_name IS NOT NULL\n"
        "       AND supplier_name NOT ILIKE '%оборонний постачальник%'\n"
        "       AND supplier_name NOT ILIKE '%приховано%'\n"
        "       AND (cpv_main LIKE '31%' OR cpv_description ILIKE '%генератор%' "
        "OR title ILIKE '%генератор%')\n"
        "     GROUP BY supplier_name ORDER BY wins DESC LIMIT 20\n"
        "   (CPV 31 і «генератори» тут — довільний приклад СИНТАКСИСУ; "
        "підставляй реальну категорію з питання користувача. Категорійну "
        "умову скомбіновано через OR за правилом 4 з тієї ж причини — "
        "неправильно вгаданий CPV-префікс інакше поверне порожній "
        "результат.)\n"
        "9. Якщо запит поєднує ЗАГАЛЬНИЙ агрегат (напр. середня сума по "
        "категорії) З пошуком ОДНОГО конкретного рекордного рядка (напр. "
        "замовник з найбільшим одноразовим платежем) — не намагайся "
        "витиснути все в один SELECT без структури, це і призводить до "
        "синтаксичних помилок. Побудуй два окремі CTE: один — скалярний "
        "агрегат, інший — ORDER BY ... LIMIT 1 для рекордного рядка, і "
        "об'єднай їх у фінальному SELECT (скалярний CTE підставляється як "
        "підзапит, без потреби в JOIN, бо кожен CTE тут повертає рівно "
        "один рядок):\n"
        "   ПРАВИЛЬНИЙ приклад СИНТАКСИСУ (товар тут — офісний папір, "
        "довільний; підставляй РЕАЛЬНУ категорію з питання користувача):\n"
        "     WITH avg_amount AS (\n"
        "       SELECT AVG(awarded_amount) AS avg_amount FROM tenders\n"
        "       WHERE cpv_main LIKE '30%'\n"
        "     ),\n"
        "     top_payment AS (\n"
        "       SELECT buyer_name, awarded_amount FROM tenders\n"
        "       WHERE cpv_main LIKE '30%' AND awarded_amount IS NOT NULL\n"
        "       ORDER BY awarded_amount DESC\n"
        "       LIMIT 1\n"
        "     )\n"
        "     SELECT\n"
        "       (SELECT avg_amount FROM avg_amount) AS average_amount,\n"
        "       top_payment.buyer_name,\n"
        "       top_payment.awarded_amount AS max_single_payment\n"
        "     FROM top_payment\n"
        "10. Виведи ТІЛЬКИ сам SQL-запит, без пояснень, без markdown-блоків, без крапки з комою в кінці."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    raw = llm_client.chat(messages=messages, model=llm_model)
    messages = messages + [{"role": "assistant", "content": raw}]
    return extract_sql(raw), messages


def validate_sql(sql: str) -> exp.Expression:
    """Parse and gate the generated SQL: exactly one statement, must be a
    SELECT/UNION, and it may only touch the whitelisted tables via
    whitelisted (non-file-reading) constructs. Raises ValueError on any
    violation — callers must not execute unvalidated SQL."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except Exception as e:
        raise ValueError(f"Не вдалося розпарсити SQL: {e}")

    if len(statements) != 1:
        raise ValueError(f"Очікувався рівно один SQL-запит, отримано {len(statements)}")

    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union)):
        raise ValueError(f"Дозволені лише SELECT-запити, отримано {type(stmt).__name__}")

    # WITH-clause CTEs are local names the query defines for itself (e.g.
    # `WITH avg_price AS (...)`), not real tables — a bare table-name
    # whitelist would reject any legitimate `FROM avg_price` reference to
    # one. Collect them so the check below doesn't misfire on the CTE
    # comparison pattern this project's SQL generator now produces.
    cte_names = {cte.alias.lower() for cte in stmt.find_all(exp.CTE) if cte.alias}
    allowed_sources = ALLOWED_TABLES | cte_names

    for tbl in stmt.find_all(exp.Table):
        # A plain named table (`FROM tenders`) has tbl.name set. Anything
        # else here — notably a table-valued function like
        # `FROM read_csv_auto('/etc/passwd')`, which DuckDB parses as a
        # Table wrapping an Anonymous func with an empty .name — is an
        # unresolvable source and must be rejected outright, not skipped.
        if not tbl.name or tbl.name.lower() not in allowed_sources:
            raise ValueError(f"Запит звертається до недозволеного джерела даних '{tbl.name or tbl.sql()}'")

    for func in stmt.find_all(exp.Func):
        func_name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()) or ""
        func_name = func_name.lower()
        if func_name in DANGEROUS_FUNCS:
            raise ValueError(f"Запит використовує недозволену функцію '{func_name}'")

    # DuckDB's LIKE is case-sensitive, but the LLM has no reliable way to
    # reproduce a text column's exact capitalization (real data here is
    # "Медичні матеріали"; the model routinely generates lowercase
    # "медичні матеріали" or similar) — a case mismatch silently returns
    # zero rows and looks exactly like "no data exists" rather than a bug.
    # Force every LIKE to ILIKE post-validation rather than depending on
    # prompt compliance to get casing right.
    for like in list(stmt.find_all(exp.Like)):
        like.replace(exp.ILike(this=like.this.copy(), expression=like.expression.copy()))

    return stmt


def execute_sql(db_path: Path, validated_stmt: exp.Expression) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(validated_stmt.sql(dialect="duckdb")).fetchdf()
    finally:
        con.close()


# Minimum comparison-group size for an average to be treated as meaningful
# for anomaly detection — matches the wording taught in generate_sql's
# apples-to-apples rule ("sample_size менше 3").
MIN_FAIR_SAMPLE_SIZE = 3


def small_sample_caveat(df: pd.DataFrame) -> str | None:
    """Deterministic backstop for the apples-to-apples fix: generate_sql
    is taught to add a `sample_size` column when computing a comparison
    average, and the narrator prompt is taught to caveat a small one —
    but prompt compliance isn't guaranteed. If the column is there and
    the sample really is small, say so regardless of what the LLM does
    with it, rather than trusting it noticed."""
    if "sample_size" not in df.columns or df.empty:
        return None
    try:
        min_n = pd.to_numeric(df["sample_size"], errors="coerce").min()
    except Exception:
        return None
    if pd.isna(min_n) or min_n >= MIN_FAIR_SAMPLE_SIZE:
        return None
    return (
        f"ℹ️ Розмір вибірки для порівняння замалий (N={int(min_n)}) — "
        "висновок про цінову аномалію на основі такого середнього може "
        "бути ненадійним.\n\n"
    )


def narrate_sql_result(llm_model: str, query: str, df: pd.DataFrame) -> str:
    max_rows = 30
    truncated = len(df) > max_rows
    table_text = df.head(max_rows).to_string(index=False)

    system_prompt = (
        "Ти — не аналітик, а гострий журналіст-розслідувач з питань "
        "антикорупції у сфері державних закупівель України. Наведи "
        "відповідь на питання користувача, спираючись ЛИШЕ на дані таблиці "
        "нижче — це реальний результат SQL-запиту до бази тендерів Prozorro. "
        "Не вигадуй чисел, яких немає в таблиці. Відповідай українською.\n\n"
        "РОЛЬ АНТИКОРУПЦІЙНОГО РОЗСЛІДУВАЧА:\n"
        "1. Якщо в таблиці є колонка із середнім/медіанним значенням поруч "
        "із окремими рядками (наприклад avg_unit_amount) — активно "
        "порівнюй кожен рядок із нею.\n"
        "   КРИТИЧНО (напрямок порівняння): «Ризик переплати» стосується "
        "ЛИШЕ рядків, де значення СУТТЄВО ВИЩЕ за середнє. Рядок, де "
        "значення НИЖЧЕ за середнє — це вигідна ціна, а НЕ переплата і "
        "НЕ аномалія; ніколи не вішай попередження на такий рядок. "
        "Порівняй кожне значення з середнім явно (вище чи нижче?) перед "
        "тим, як щось позначати.\n"
        "   ВАЖЛИВО: якщо в таблиці є колонка sample_size — це кількість "
        "товарів, які увійшли в розрахунок середнього. Якщо sample_size "
        "менше 3 — вибірка замала для чесного висновку про аномалію; "
        "прямо скажи це користувачу («недостатньо схожих товарів для "
        "порівняння») ЗАМІСТЬ того, щоб позначати цінову різницю як "
        "переплату.\n"
        "2. Якщо один постачальник (supplier_name) повторюється в кількох "
        "рядках або явно домінує за кількістю/сумою — це теж аномалія.\n"
        "   ВИНЯТОК: значення supplier_name = «Оборонний постачальник» "
        "(або «Приховано») — це НЕ назва реальної компанії, а офіційне "
        "маскування Prozorro для оборонних закупівель, засекречених з "
        "міркувань нацбезпеки. Під цим написом об'єднані десятки РІЗНИХ "
        "реальних постачальників. НІКОЛИ не позначай його як монополію чи "
        "домінування одного постачальника — якщо воно з'являється в "
        "таблиці, поясни користувачу, що ці дані замасковані, а не "
        "видавай висновок про одну компанію.\n"
        "3. Кожна аномалія з пунктів 1-2 ОБОВ'ЯЗКОВО оформлюється в одному "
        "реченні за такою структурою, без винятків:\n"
        "   «⚠️ Можлива аномалія / Ризик переплати: [конкретне ім'я "
        "постачальника/замовника з таблиці] — [конкретні числа з таблиці, "
        "напр. 24 перемоги проти 19 у наступного, або 71 500 грн проти "
        "середньої 1 932 грн] — [коротке пояснення механізму ризику: "
        "монополія, слабка конкуренція, тендер під конкретного "
        "постачальника, суттєва переплата тощо].»\n"
        "   Фраза «⚠️ Можлива аномалія / Ризик переплати» — ОБОВ'ЯЗКОВА "
        "частина цього речення, а не окрема опція; ніколи не пиши "
        "висновок про аномалію (домінування, переплату, слабку "
        "конкуренцію) без неї. Символ ⚠️ на початку — теж обов'язкова "
        "частина фрази, а не прикраса; НІКОЛИ не пропускай і не заміняй "
        "його.\n"
        "4. Якщо в таблиці НЕМАЄ жодної аномалії за критеріями 1-2 — не "
        "вживай фразу і не вигадуй проблему.\n"
        "5. Для кожного факту (аномального чи ні) посилайся на конкретні "
        "імена/tender_id/title з таблиці — ніколи на узагальнення на "
        "кшталт «один постачальник» чи «деякі тендери».\n"
        "   КРИТИЧНО: якщо в таблиці НЕМАЄ колонки supplier_name/"
        "buyer_name (напр. запит рахував лише ціни по товарах, без "
        "постачальників) — НІКОЛИ не вигадуй назву компанії, щоб "
        "'заповнити' речення. Посилайся ЛИШЕ на tender_id/description, "
        "які РЕАЛЬНО є в таблиці. Вигадана назва компанії — це "
        "найгірший тип помилки тут: вона звучить правдоподібно, але "
        "стосується когось, хто, можливо, взагалі не причетний.\n"
        "6. Формат: гранично стисло, буліт-пойнти. НІКОЛИ не починай "
        "канцелярсько — заборонені фрази на кшталт «На підставі наданого "
        "контексту...», «Based on the provided context...» тощо. Одразу факти."
    )
    # Targeted, per-query reinforcement rather than relying on the general
    # system prompt alone: observed live that when a table has no
    # supplier/buyer column at all, the model can still invent a
    # plausible-sounding company name to "complete" the anomaly sentence.
    # Stating the concrete condition close to the actual data is more
    # reliable than a general rule stated once, far away, in the system
    # prompt — same reasoning as putting the CTE examples' reminders
    # right next to the example itself rather than only at the top.
    no_entity_warning = ""
    if "supplier_name" not in df.columns and "buyer_name" not in df.columns:
        no_entity_warning = (
            "\n\nУВАГА: у цій таблиці НЕМАЄ колонки з постачальником чи "
            "замовником — НЕ називай жодну компанію чи установу, її "
            "просто немає в цих даних."
        )
    prompt = (
        f"Питання: {query}\n\n"
        f"Результат SQL-запиту{' (показано перші ' + str(max_rows) + ' рядків)' if truncated else ''}:\n"
        f"{table_text}{no_entity_warning}\n\nВідповідь:"
    )
    raw = llm_client.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        model=llm_model,
    )
    answer = strip_fabricated_links(raw)
    caveat = small_sample_caveat(df)
    return (caveat + answer) if caveat else answer


MAX_SQL_ATTEMPTS = 3


def answer_sql_query(db_path: Path, llm_model: str, query: str) -> str:
    schema_desc = get_schema_description(db_path)

    messages = None
    sql = ""
    last_error = None

    for attempt in range(1, MAX_SQL_ATTEMPTS + 1):
        if last_error is not None:
            # Feed the model its own failed SQL plus the actual error back
            # as the next conversational turn, so it can self-correct
            # instead of repeating the same mistake — rather than just
            # giving up on the first failure.
            messages = messages + [{
                "role": "user",
                "content": (
                    f"Цей SQL впав з помилкою: {last_error}\n\n"
                    "Виправ його і поверни ЛИШЕ виправлений SQL-запит, "
                    "без пояснень і без markdown-блоків."
                ),
            }]
        sql, messages = generate_sql(llm_model, query, schema_desc, messages=messages)

        try:
            validated_stmt = validate_sql(sql)
        except ValueError as e:
            print(f"[SQL DEBUG] Attempt {attempt}/{MAX_SQL_ATTEMPTS} rejected by validation gate ({e}):\n{sql}")
            logger.warning("Attempt %d: generated SQL rejected by validation gate: %s", attempt, e)
            last_error = str(e)
            continue

        validated_sql = validated_stmt.sql(dialect="duckdb")
        print(f"[SQL DEBUG] Attempt {attempt}/{MAX_SQL_ATTEMPTS} executing:\n{validated_sql}")

        try:
            df = execute_sql(db_path, validated_stmt)
        except Exception as e:
            print(f"[SQL DEBUG] Attempt {attempt}/{MAX_SQL_ATTEMPTS} execution failed ({type(e).__name__}: {e}):\n{validated_sql}")
            logger.warning("Attempt %d: SQL execution failed: %s", attempt, e)
            last_error = f"{type(e).__name__}: {e}"
            continue

        if df.empty:
            return "Запит не повернув жодного результату — можливо, таких даних немає у завантаженому наборі."

        return narrate_sql_result(llm_model, query, df)

    # The raw SQL/error detail is developer-debugging output only (see
    # [SQL DEBUG] above) — it never belongs in a message an end user sees.
    logger.warning("Gave up on SQL generation for %r after %d attempts", query, MAX_SQL_ATTEMPTS)
    return (
        f"Не вдалося сформувати робочий SQL-запит для цього питання навіть "
        f"після {MAX_SQL_ATTEMPTS} спроб. Спробуйте переформулювати його простіше."
    )


# --------------------------------------------------------------------------
# Vector (qualitative) path
# --------------------------------------------------------------------------

def retrieve(collection, embedder: Embedder, query: str, top_k: int) -> list[dict]:
    query_vec = embedder.embed_query(query)
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits


def build_context(hits: list[dict]) -> tuple[str, list[dict]]:
    """Split hits into relevant (below threshold) vs all, and render the
    relevant ones into a context block. Returns (context_text, relevant_hits)."""
    relevant = [h for h in hits if h["distance"] <= RELEVANCE_THRESHOLD]
    context_text = "\n---\n".join(h["text"] for h in relevant)
    return context_text, relevant


# Neither the chunk text built in load_chroma.py nor a DuckDB result table
# ever contains a URL — Prozorro tender data has no such field. So any link
# the LLM produces is fabricated by construction, no matter what the
# prompt says. Don't rely on prompt compliance alone (the same reasoning
# as validate_sql's read-only gate): strip it deterministically.
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((?:https?://|www\.)[^)]*\)")
_BARE_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+")


def strip_fabricated_links(text: str) -> str:
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _BARE_URL_RE.sub("", text)
    return text


# Observed live: even with an explicit prompt instruction not to, the model
# sometimes answers a technical-specs question (weight, flight range, etc.)
# by pulling plausible-sounding numbers for a named product (e.g. "DJI
# Mavic 3") from its own general/training knowledge rather than the actual
# retrieved chunk — which never contains those numbers (see
# build_chunk_text in load_chroma.py: title/buyer/CPV/budget/status only,
# no physical specs). This is a measurement-shaped number, so it's not
# caught by strip_fabricated_links. Prompt reinforcement alone didn't make
# this reliable (same diminishing-returns pattern as the anomaly-flag
# emoji), so — same principle as the SQL validation gate — add a
# deterministic check instead of trusting compliance: flag any
# measurement-looking number in the answer that doesn't appear verbatim in
# the retrieved context, rather than silently presenting it as grounded.
_SPEC_MEASUREMENT_RE = re.compile(
    r"\d[\d\s.,]*\s*(?:кг|км/год|км|м/с|м\b|грам\w*|г\.(?!рн)|метр\w*|"
    r"годин\w*|хвилин\w*|хв\.?|вт\b|ват\w*|вольт\w*)",
    re.IGNORECASE,
)


def flag_ungrounded_measurements(answer: str, context_text: str) -> str:
    found = {m.group().strip() for m in _SPEC_MEASUREMENT_RE.finditer(answer)}
    ungrounded = sorted(m for m in found if m not in context_text)
    if not ungrounded:
        return answer
    warning = (
        "⚠️ Увага: конкретні цифри в цій відповіді ("
        + ", ".join(ungrounded)
        + ") не знайдені дослівно в базі тендерів — модель могла підставити "
        "їх із загальних знань про товар, а не з реальних даних цього "
        "тендера. Перевірте оригінальну документацію на Prozorro перед тим, "
        "як покладатися на ці цифри.\n\n"
    )
    return warning + answer


# The Telegram bot renders answers with legacy Markdown (parse_mode="Markdown")
# so tender-number citations can become clickable links. Verified against
# the live site (not assumed, given this project's history with a
# fabricated Prozorro domain): https://prozorro.gov.ua/tender/<tender_number>
# returns the real tender page for a known-real tender_number.
PROZORRO_TENDER_URL_TEMPLATE = "https://prozorro.gov.ua/tender/{tender_number}"
_TENDER_ID_RE = re.compile(r"\bUA-\d{4}-\d{2}-\d{2}-\d{6}-[A-Za-zА-Яа-яЇїІіЄєҐґ]\b")
# Legacy Telegram Markdown only treats these four characters as syntax
# (unlike MarkdownV2, which requires escaping ~20 characters everywhere) —
# far lower risk of an unrelated character in a company name or LLM
# sentence accidentally breaking the parser.
_MD_V1_SPECIAL_RE = re.compile(r"([_*`\[])")


def escape_markdown_v1(text: str) -> str:
    return _MD_V1_SPECIAL_RE.sub(r"\\\1", text)


def linkify_known_tenders(text: str, known_ids: set[str]) -> str:
    """Escape `text` for Telegram's legacy Markdown, turning any
    tender-number-shaped token into a clickable Prozorro link — but ONLY
    if that exact number is one we actually retrieved (`known_ids`).
    Never trust the LLM to reproduce a 20+ character ID correctly on its
    own: a single wrong digit would otherwise silently produce a dead or
    misleading link. An ID-shaped token that isn't in `known_ids` (model
    garbled it, or invented one) is left as escaped plain text instead —
    visible, but not clickable, rather than a broken link."""
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


def answer_vector_query(collection, embedder: Embedder, llm_model: str, query: str, top_k: int) -> str:
    hits = retrieve(collection, embedder, query, top_k)
    context_text, relevant = build_context(hits)

    if not relevant:
        # Don't even call the LLM with irrelevant context — that's how you
        # get confident-sounding answers built on medical tenders when the
        # user asked about drones. Be explicit about the miss instead.
        closest = hits[0]["distance"] if hits else None
        logger.warning(
            "No results under relevance threshold %.2f (closest: %s) — refusing to answer from context",
            RELEVANCE_THRESHOLD, f"{closest:.4f}" if closest is not None else "n/a",
        )
        return (
            "У завантажених даних не знайшлося тендерів, достатньо релевантних "
            "цьому запиту. Можливо, потрібної категорії немає у вашому наразі "
            "завантаженому наборі даних, або варто розширити пошук (Крок 1 — "
            "перевірте фільтри CPV/ключових слів чи діапазон дат екстракції)."
        )

    prompt = f"Контекст:\n{context_text}\n\nПитання: {query}\n\nВідповідь:"

    raw = llm_client.chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=llm_model,
    )
    answer = strip_fabricated_links(raw)
    answer = flag_ungrounded_measurements(answer, context_text)

    known_ids = {h["metadata"].get("tender_number") for h in relevant if h["metadata"].get("tender_number")}
    answer = linkify_known_tenders(answer, known_ids)

    def _source_line(h: dict) -> str:
        tid = h["metadata"].get("tender_number")
        label = f"[{tid}]({PROZORRO_TENDER_URL_TEMPLATE.format(tender_number=tid)})" if tid else "?"
        title = escape_markdown_v1(str(h["metadata"].get("title", "?")))
        return f"  - {label} (dist={h['distance']:.4f}): {title}"

    sources = "\n".join(_source_line(h) for h in relevant)
    return f"{answer}\n\nДжерела:\n{sources}"


# --------------------------------------------------------------------------
# Router entry point
# --------------------------------------------------------------------------

def answer_query(
    collection, embedder: Embedder, db_path: Path, llm_model: str, query: str, top_k: int
) -> str:
    if is_conversational_followup(query):
        logger.info("Detected a conversational follow-up — bot is stateless, returning fallback instead of guessing")
        return FOLLOWUP_FALLBACK_TEXT

    # Checked before the generic router: these are specific, high-value
    # corruption-investigation shapes (tender splitting, local monopoly,
    # apples-to-apples overpricing) that get a hardcoded SQL template
    # instead of free-form generation — see investigations.py for why.
    investigation_kind = classify_investigation(query)
    if investigation_kind:
        logger.info("Routed query to investigation template '%s'", investigation_kind)
        return run_investigation(investigation_kind, db_path, llm_model, query)

    route = classify_query(query)
    logger.info("Routed query to '%s' path", route)
    if route == "sql":
        return answer_sql_query(db_path, llm_model, query)
    return answer_vector_query(collection, embedder, llm_model, query, top_k)


def interactive_loop(collection, embedder: Embedder, db_path: Path, llm_model: str, top_k: int):
    print("Prozorro Defense AI Explorer — hybrid RAG mode. Ctrl+C or 'exit' to quit.\n")
    while True:
        try:
            query = input("Питання> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in ("exit", "quit"):
            break
        print()
        print(answer_query(collection, embedder, db_path, llm_model, query, top_k))
        print()


def main():
    parser = argparse.ArgumentParser(description="Hybrid SQL + vector RAG over Prozorro tenders via Ollama.")
    parser.add_argument("--chroma-path", type=Path, default=Path("chroma_db"))
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="DuckDB file for the SQL routing path")
    parser.add_argument("--embed-model", default=DEFAULT_MODEL, help="Must match the model used in load_chroma.py")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Ollama model tag, e.g. qwen2.5:14b-instruct")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", type=str, default=None, help="Single query; omit for interactive mode")
    args = parser.parse_args()

    if not args.chroma_path.exists():
        logger.error("Chroma path %s does not exist — run load_chroma.py first", args.chroma_path)
        sys.exit(1)

    if not args.db_path.exists():
        logger.error("DuckDB file %s does not exist — run transform_prozorro.py first", args.db_path)
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(args.chroma_path))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        logger.error("Collection '%s' not found at %s — run load_chroma.py first", COLLECTION_NAME, args.chroma_path)
        sys.exit(1)

    embedder = Embedder(args.embed_model)

    # Fail fast with a clear message if the configured LLM provider isn't
    # reachable, rather than a raw exception on the first query.
    try:
        llm_client.health_check()
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)

    if args.query:
        print(answer_query(collection, embedder, args.db_path, args.llm_model, args.query, args.top_k))
    else:
        interactive_loop(collection, embedder, args.db_path, args.llm_model, args.top_k)


if __name__ == "__main__":
    main()

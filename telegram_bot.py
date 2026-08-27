"""
telegram_bot.py
Step 5 of the Prozorro Defense AI Explorer pipeline — Telegram interface.

Wraps the existing hybrid RAG engine (rag_query.py) in a Telegram bot built
on aiogram v3. The embedding model and ChromaDB collection are loaded once
at startup and reused across requests; each incoming message runs the
(synchronous) RAG pipeline in a worker thread via asyncio.to_thread so it
doesn't block the bot's event loop.

UI: /start shows a persistent reply keyboard with four shortcuts. The two
mode buttons ("Аналітика ринку" / "Пошук тендерів") set a one-shot forced
route for the user's next free-text message, bypassing the keyword router's
guesswork; "Перевірка аномалій" runs a canned supplier-concentration scan
immediately. /stats reports live row counts from DuckDB. /analyze <tender_id>
looks up one tender directly and compares its price to its category average
— deterministically, in Python, with no LLM call, since that comparison
doesn't need free-text generation and a direct lookup is more reliable than
asking a model to reproduce it.

"Пошук схем (Аудит)" (or /schemes) opens an inline keyboard for three
hardcoded corruption-investigation templates from investigations.py —
tender splitting, local-monopoly favoritism, apples-to-apples overpricing —
instead of free-form SQL generation for these specific, high-value query
shapes. The splitting scan runs immediately (no parameter needed); the
other two prompt for a buyer/item and pick up the next free-text reply the
same way the SQL/RAG mode buttons do.

An AsyncIOScheduler (not a separate OS thread — it runs on this same
asyncio event loop, the way the rest of this file already prefers)
triggers update_data.run_update() daily at UPDATE_SCHEDULE_HOUR:MINUTE
(.env, default 3:00 AM) via asyncio.to_thread, so the nightly sync can't
block the bot from answering messages while it runs. See DEPLOY.md for
the equivalent system-crontab alternative if you'd rather not run the
scheduler inside the bot process at all.

Install:
    pip install aiogram python-dotenv apscheduler
    (chromadb / sentence-transformers / duckdb / ollama / sqlglot already
    installed from earlier steps)

.env setup:
    Copy .env.example to .env in this directory and fill in your bot token:
        cp .env.example .env
    Get a token from @BotFather on Telegram (send /newbot, follow the
    prompts), then set:
        TELEGRAM_BOT_TOKEN=123456789:AAExampleTokenFromBotFather

Usage:
    python telegram_bot.py
"""

import asyncio
import logging
import os
import re
from pathlib import Path

import chromadb
import duckdb
from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from dotenv import load_dotenv

from load_chroma import COLLECTION_NAME, DEFAULT_MODEL, Embedder
from rag_query import (
    DEFAULT_DB_PATH,
    DEFAULT_LLM_MODEL,
    answer_query,
    answer_sql_query,
    answer_vector_query,
)
from investigations import run_investigation
import llm_client
import update_data
from apscheduler.events import EVENT_JOB_MISSED
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prozorro_bot")

TOP_K = 5
TELEGRAM_MSG_LIMIT = 4096
PLACEHOLDER_TEXT = "⏳ Шукаю в базі та аналізую тендери. Це може зайняти кілька секунд..."
# Simple heuristic ratio for /analyze's price-anomaly check — same 1.3x
# margin the LLM prompts use elsewhere in this project, kept consistent
# rather than inventing a second threshold. Not a statistical test, just a
# quick "is this notably above the category average" signal.
PRICE_ANOMALY_RATIO = 1.3

BTN_SQL = "📊 Аналітика ринку (SQL)"
BTN_RAG = "🔎 Пошук тендерів (RAG)"
BTN_ANOMALY = "🚨 Перевірка аномалій"
BTN_SCHEMES = "🚨 Пошук схем (Аудит)"

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text=BTN_SQL), KeyboardButton(text=BTN_RAG)],
        [KeyboardButton(text=BTN_ANOMALY)],
        [KeyboardButton(text=BTN_SCHEMES)],
    ],
    resize_keyboard=True,
)

# callback_data for each inline button — matched by handle_scheme_callback
# via a "scheme:" prefix filter.
SCHEMES_KEYBOARD = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="🔍 Перевірити дроблення тендерів", callback_data="scheme:splitting")],
    [InlineKeyboardButton(text="💰 Перевірити переплати", callback_data="scheme:overpricing")],
    [InlineKeyboardButton(text="🏢 Аналіз монополій", callback_data="scheme:monopoly")],
])

ANOMALY_SCAN_QUERY = (
    "Хто з постачальників найчастіше виграє тендери в базі, і чи є серед "
    "них цінові аномалії або підозріло висока концентрація перемог?"
)
SPLITTING_SCAN_QUERY = "Чи є ознаки дроблення тендерів у базі?"

# Prompts shown after tapping a scheme button that needs a parameter the
# button tap itself can't supply — the user's next free-text reply is
# picked up by handle_query via _pending_mode, same mechanism as the
# BTN_SQL/BTN_RAG mode buttons.
SCHEME_PARAM_PROMPTS = {
    "monopoly": "Вкажіть замовника, якого перевірити на монополію постачальників:",
    "overpricing": "Вкажіть модель чи тип товару, який перевірити на переплату:",
}

WELCOME_TEXT = (
    "Привіт! Я аналізую державні закупівлі України, пов'язані з обороною та "
    "волонтерською підтримкою ЗСУ, на основі відкритих даних Prozorro.\n\n"
    "Скористайтесь кнопками нижче або пишіть напряму:\n"
    f"• {BTN_SQL} — підрахунки, суми, рейтинги постачальників\n"
    f"• {BTN_RAG} — пошук конкретних тендерів за темою\n"
    f"• {BTN_ANOMALY} — миттєва перевірка ринку на аномалії\n"
    f"• {BTN_SCHEMES} — цілеспрямований аудит: дроблення тендерів, "
    "монополії постачальників, переплати за конкретну модель товару\n\n"
    "Команди:\n"
    "/stats — швидка статистика бази\n"
    "/analyze <номер тендера> — детальний розбір одного тендера\n"
    "/schemes — те саме, що кнопка «Пошук схем»\n\n"
    "Я відповідаю лише на основі реальних даних, завантажених у базу, і "
    "чесно скажу, якщо їх недостатньо для відповіді — без вигадування "
    "цифр чи фактів."
)

# One-shot forced route per chat: set by a mode button, consumed (and
# cleared) by the very next free-text message from that chat. Values are
# "sql", "vector", or "investigation:<kind>" for the /schemes flow.
_pending_mode: dict[int, str] = {}


def split_message(text: str, limit: int = TELEGRAM_MSG_LIMIT) -> list[str]:
    """Telegram rejects messages over 4096 chars — split on line breaks
    where possible so a long answer + sources block still arrives readable
    rather than getting rejected outright."""
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    chunks.append(text)
    return chunks


# rag_query.py's vector-path answers now contain legacy-Markdown links
# (clickable tender numbers). Free-form LLM/DB text can still contain a
# stray _/*/`/[  that breaks Telegram's parser — never let a formatting
# edge case silently swallow a response: fall back to plain text (turning
# a link into "text (url)") rather than raising.
_MD_LINK_TO_PLAIN_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_MD_ESCAPE_RE = re.compile(r"\\([_*`\[])")


def _markdown_to_plain(text: str) -> str:
    text = _MD_LINK_TO_PLAIN_RE.sub(r"\1 (\2)", text)
    return _MD_ESCAPE_RE.sub(r"\1", text)


async def send_markdown(message: Message, text: str) -> None:
    try:
        await message.answer(text, parse_mode="Markdown")
    except TelegramBadRequest:
        logger.warning("Markdown parse failed on send — falling back to plain text")
        await message.answer(_markdown_to_plain(text))


async def edit_markdown(placeholder: Message, text: str) -> None:
    try:
        await placeholder.edit_text(text, parse_mode="Markdown")
    except TelegramBadRequest:
        logger.warning("Markdown parse failed on edit — falling back to plain text")
        await placeholder.edit_text(_markdown_to_plain(text))


async def cmd_start(message: Message) -> None:
    await message.answer(WELCOME_TEXT, reply_markup=MAIN_KEYBOARD)


async def cmd_stats(message: Message, db_path: Path) -> None:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        n_tenders, min_date, max_date = con.execute(
            "SELECT COUNT(*), MIN(date_created), MAX(date_created) FROM tenders"
        ).fetchone()
        n_items = con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        n_awarded = con.execute(
            "SELECT COUNT(*) FROM tenders WHERE awarded_amount IS NOT NULL"
        ).fetchone()[0]
    finally:
        con.close()

    date_range = (
        f"{min_date:%Y-%m-%d} — {max_date:%Y-%m-%d}" if min_date and max_date else "н/д"
    )
    await message.answer(
        "📊 Статистика бази:\n"
        f"— В базі: {n_tenders} тендерів, {n_items} позицій закупівель\n"
        f"— З відомою сумою контракту: {n_awarded}\n"
        f"— Період: {date_range}"
    )


async def cmd_analyze(message: Message, command: CommandObject, db_path: Path) -> None:
    tender_id = (command.args or "").strip()
    if not tender_id:
        await message.answer(
            "Вкажіть номер тендера: /analyze UA-2024-01-01-000110-a"
        )
        return

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute(
            "SELECT tender_number, title, status, buyer_name, buyer_region, "
            "supplier_name, cpv_main, cpv_description, budget_amount, "
            "budget_currency, awarded_amount, awarded_currency "
            "FROM tenders WHERE tender_number = ?",
            [tender_id],
        ).fetchone()

        if row is None:
            await message.answer(f"Тендер {tender_id!r} не знайдено в базі.")
            return

        (tender_number, title, status, buyer_name, buyer_region, supplier_name,
         cpv_main, cpv_description, budget_amount, budget_currency,
         awarded_amount, awarded_currency) = row

        avg_row = con.execute(
            "SELECT AVG(awarded_amount) FROM tenders "
            "WHERE cpv_main = ? AND awarded_amount IS NOT NULL AND tender_number != ?",
            [cpv_main, tender_number],
        ).fetchone()
        category_avg = avg_row[0] if avg_row else None
    finally:
        con.close()

    lines = [
        f"🔍 {tender_number}",
        f"Назва: {title or 'н/д'}",
        f"Статус: {status or 'н/д'}",
        f"Замовник: {buyer_name or 'н/д'} ({buyer_region or 'н/д'})",
        f"Категорія (CPV {cpv_main or 'н/д'}): {cpv_description or 'н/д'}",
    ]

    if supplier_name and ("оборонний постачальник" in supplier_name.lower() or "приховано" in supplier_name.lower()):
        lines.append(
            "Постачальник: дані замасковані Prozorro з міркувань нацбезпеки "
            "(це не назва конкретної компанії)"
        )
    else:
        lines.append(f"Постачальник: {supplier_name or 'н/д'}")

    if budget_amount is not None:
        lines.append(f"Орієнтовний бюджет: {budget_amount:,.2f} {budget_currency or ''}".replace(",", " "))
    if awarded_amount is not None:
        lines.append(f"Сума контракту: {awarded_amount:,.2f} {awarded_currency or ''}".replace(",", " "))

        if category_avg and category_avg > 0:
            ratio = awarded_amount / category_avg
            lines.append(f"Середня сума в категорії: {category_avg:,.2f}".replace(",", " "))
            if ratio >= PRICE_ANOMALY_RATIO:
                lines.append(
                    f"⚠️ Можлива аномалія / Ризик переплати: сума в {ratio:.1f} "
                    "раза вища за середню в цій категорії."
                )
    else:
        lines.append("Сума контракту: невідома (тендер ще не завершено або не розкрито)")

    await message.answer("\n".join(lines))


async def handle_mode_button(message: Message) -> None:
    text = (message.text or "").strip()
    mode = "sql" if text == BTN_SQL else "vector"
    _pending_mode[message.chat.id] = mode
    prompt = (
        "Введіть аналітичне питання (напр. «скільки тендерів на генератори "
        "створено у 2024 році» або «хто найчастіше виграє тендери»):"
        if mode == "sql"
        else "Введіть тему для пошуку (напр. «дрони для розвідки»):"
    )
    await message.answer(prompt)


async def handle_anomaly_button(message: Message, db_path: Path, llm_model: str) -> None:
    placeholder = await message.answer(PLACEHOLDER_TEXT)
    await message.bot.send_chat_action(message.chat.id, "typing")
    try:
        answer = await asyncio.to_thread(answer_sql_query, db_path, llm_model, ANOMALY_SCAN_QUERY)
    except Exception:
        logger.exception("Anomaly scan failed")
        await placeholder.edit_text("Сталася помилка під час перевірки аномалій. Спробуйте ще раз.")
        return

    chunks = split_message(answer)
    await edit_markdown(placeholder, chunks[0])
    for chunk in chunks[1:]:
        await send_markdown(message, chunk)


async def cmd_schemes(message: Message) -> None:
    await message.answer("Оберіть тип перевірки:", reply_markup=SCHEMES_KEYBOARD)


async def handle_scheme_callback(callback: CallbackQuery, db_path: Path, llm_model: str) -> None:
    # Acknowledge immediately so Telegram stops showing the loading spinner
    # on the tapped button, independent of how long the investigation takes.
    await callback.answer()
    kind = (callback.data or "").split(":", 1)[-1]

    if kind == "splitting":
        # No parameter needed — the splitting template scans the whole
        # market by default — so this can run immediately, same pattern
        # as the BTN_ANOMALY canned scan.
        placeholder = await callback.message.answer(PLACEHOLDER_TEXT)
        await callback.bot.send_chat_action(callback.message.chat.id, "typing")
        try:
            answer = await asyncio.to_thread(
                run_investigation, "splitting", db_path, llm_model, SPLITTING_SCAN_QUERY
            )
        except Exception:
            logger.exception("Splitting investigation failed")
            await placeholder.edit_text("Сталася помилка під час перевірки. Спробуйте ще раз.")
            return
        chunks = split_message(answer)
        await edit_markdown(placeholder, chunks[0])
        for chunk in chunks[1:]:
            await send_markdown(callback.message, chunk)
        return

    # monopoly/overpricing need a parameter (buyer or item) that a button
    # tap alone can't supply — ask for it and let handle_query pick up the
    # user's next free-text reply via _pending_mode.
    _pending_mode[callback.message.chat.id] = f"investigation:{kind}"
    await callback.message.answer(SCHEME_PARAM_PROMPTS[kind])


async def handle_query(message: Message, collection, embedder: Embedder, db_path: Path, llm_model: str) -> None:
    query = (message.text or "").strip()
    if not query:
        return

    forced_mode = _pending_mode.pop(message.chat.id, None)

    # Send a placeholder immediately — local LLM generation can take tens
    # of seconds, and a bare "typing..." indicator alone reads as stalled.
    # Edit this same message in place once the real answer is ready,
    # rather than leaving the placeholder and sending a second message.
    placeholder = await message.answer(PLACEHOLDER_TEXT)
    await message.bot.send_chat_action(message.chat.id, "typing")
    try:
        # answer_query/answer_sql_query/answer_vector_query (and the
        # Ollama/DuckDB calls under them) are synchronous and can take tens
        # of seconds — run off the event loop so the bot keeps responding
        # to other chats/updates while it works.
        if forced_mode == "sql":
            answer = await asyncio.to_thread(answer_sql_query, db_path, llm_model, query)
        elif forced_mode == "vector":
            answer = await asyncio.to_thread(
                answer_vector_query, collection, embedder, llm_model, query, TOP_K
            )
        elif forced_mode and forced_mode.startswith("investigation:"):
            kind = forced_mode.split(":", 1)[1]
            answer = await asyncio.to_thread(run_investigation, kind, db_path, llm_model, query)
        else:
            answer = await asyncio.to_thread(
                answer_query, collection, embedder, db_path, llm_model, query, TOP_K
            )
    except Exception:
        logger.exception("Failed to answer query: %r", query)
        await placeholder.edit_text(
            "Сталася помилка під час обробки запиту. Спробуйте ще раз або "
            "переформулюйте питання."
        )
        return

    chunks = split_message(answer)
    await edit_markdown(placeholder, chunks[0])
    for chunk in chunks[1:]:
        await send_markdown(message, chunk)


async def scheduled_update_job() -> None:
    """The nightly incremental sync (see update_data.py). Runs in a worker
    thread via asyncio.to_thread — same reasoning as every RAG call in
    this file — so a multi-minute update never blocks the bot from
    answering messages while it runs. Exceptions are caught and logged
    here rather than left to propagate: an uncaught exception in an
    APScheduler job is swallowed by the scheduler anyway (it just won't
    fire again until the next scheduled time), so logging it ourselves is
    the only way to actually notice a failed nightly run.

    This alone is NOT sufficient to trust the nightly sync, though — see
    _on_job_missed and the startup staleness check in main(): a real
    Docker Desktop host sleep silently swallowed a scheduled run with
    zero log trace (confirmed live, and reproduced deliberately: an
    AsyncIOScheduler cron job whose fire time is missed by more than its
    misfire_grace_time is dropped without even the default APScheduler
    misfire WARNING, if the event loop's own wakeup callback never gets
    re-armed across the pause). The generous misfire_grace_time below
    reduces how often this happens; the startup staleness check is what
    actually recovers from it when it does."""
    logger.info("Starting scheduled incremental data update...")
    try:
        result = await asyncio.to_thread(update_data.run_update)
        logger.info("Scheduled update finished: %s", result)
    except Exception:
        logger.exception("Scheduled incremental update failed")


def _on_job_missed(event) -> None:
    logger.error(
        "Scheduled job %r MISSED its run time (%s) — likely the process/host "
        "was paused (e.g. sleep) past misfire_grace_time. It will attempt "
        "its next scheduled time; the startup staleness check in main() is "
        "the real backstop if this keeps happening.",
        event.job_id, event.scheduled_run_time,
    )


async def main() -> None:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not set. Copy .env.example to .env and fill in your bot token."
        )

    chroma_path = Path(os.getenv("CHROMA_PATH", "chroma_db"))
    db_path = Path(os.getenv("DB_PATH", str(DEFAULT_DB_PATH)))
    llm_model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)

    if not chroma_path.exists():
        raise RuntimeError(f"Chroma path {chroma_path} does not exist — run load_chroma.py first")
    if not db_path.exists():
        raise RuntimeError(f"DuckDB file {db_path} does not exist — run transform_prozorro.py first")

    logger.info("Loading embedding model and Chroma collection (once, at startup)...")
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(COLLECTION_NAME)
    embedder = Embedder(DEFAULT_MODEL)

    llm_client.health_check()

    bot = Bot(token=token)
    dp = Dispatcher()
    dp.message.register(cmd_start, CommandStart())
    dp.message.register(cmd_stats, Command("stats"))
    dp.message.register(cmd_analyze, Command("analyze"))
    dp.message.register(cmd_schemes, Command("schemes"))
    # Button handlers must be registered before the generic text handler —
    # aiogram dispatches to the first handler whose filters match, and
    # these exact-text filters would otherwise never be reached.
    dp.message.register(handle_mode_button, F.text.in_({BTN_SQL, BTN_RAG}))
    dp.message.register(handle_anomaly_button, F.text == BTN_ANOMALY)
    dp.message.register(cmd_schemes, F.text == BTN_SCHEMES)
    dp.message.register(handle_query, F.text)
    dp.callback_query.register(handle_scheme_callback, F.data.startswith("scheme:"))

    update_hour = int(os.getenv("UPDATE_SCHEDULE_HOUR", "3"))
    update_minute = int(os.getenv("UPDATE_SCHEDULE_MINUTE", "0"))
    scheduler = AsyncIOScheduler()
    scheduler.add_listener(_on_job_missed, EVENT_JOB_MISSED)
    scheduler.add_job(
        scheduled_update_job, "cron", hour=update_hour, minute=update_minute,
        # Generous on purpose: a nightly sync running an hour late (e.g.
        # after a host sleep/pause) is fine; running it not at all,
        # silently, is not. 1s is APScheduler's default, which is what
        # let a real Docker Desktop host-sleep swallow a run with zero
        # log trace during testing.
        misfire_grace_time=3600,
    )
    scheduler.start()
    logger.info("Nightly incremental update scheduled for %02d:%02d", update_hour, update_minute)

    # Startup catch-up: don't just trust the cron trigger to have fired
    # reliably while nothing was watching (see scheduled_update_job's
    # docstring) — if the data is already stale enough that a nightly run
    # was clearly missed, run one now instead of waiting for the next
    # scheduled tick. Fire-and-forget via create_task: run_update() can
    # take tens of minutes, and shouldn't delay the bot from starting to
    # poll and answer messages.
    stale_threshold_hours = float(os.getenv("UPDATE_STALE_THRESHOLD_HOURS", "30"))
    hours_stale = update_data.hours_since_last_sync(db_path)
    if hours_stale is not None and hours_stale > stale_threshold_hours:
        logger.warning(
            "Data is %.1fh stale (> %.1fh threshold) — a scheduled sync was "
            "likely missed. Running a catch-up update now.",
            hours_stale, stale_threshold_hours,
        )
        asyncio.create_task(scheduled_update_job())

    logger.info("Bot starting polling...")
    await dp.start_polling(bot, collection=collection, embedder=embedder, db_path=db_path, llm_model=llm_model)


if __name__ == "__main__":
    asyncio.run(main())

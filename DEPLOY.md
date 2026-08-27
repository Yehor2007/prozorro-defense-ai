# Production deployment

## 1. Incremental data updates

`update_data.py` is the nightly-safe counterpart to `extract_prozorro.py`'s
one-time historical backfill — see its module docstring for the full design
(resume cursor, upsert-by-merge, incremental Chroma re-embedding). Two ways
to run it on a schedule; pick one, not both:

**Option A — built in (default).** `telegram_bot.py` already starts an
`AsyncIOScheduler` job at startup (see `scheduled_update_job()`), firing
daily at `UPDATE_SCHEDULE_HOUR:UPDATE_SCHEDULE_MINUTE` (`.env`, default
`3:00`). Nothing to configure beyond those two optional `.env` values —
this is what runs when you `docker compose up -d`.

**Option B — system crontab**, if you'd rather the update run as a fully
separate process (independent logs, independent failure domain from the
bot, works even if the bot container is down):

```cron
# Prozorro incremental update — daily at 3:00 AM
0 3 * * * cd /path/to/prozorro-defense-ai && /path/to/venv/bin/python update_data.py >> update_data.log 2>&1
```

If you use option B, remove/ignore the scheduler inside `telegram_bot.py`
by setting `UPDATE_SCHEDULE_HOUR`/`MINUTE` doesn't disable it — there's no
env flag to turn the built-in scheduler off short of not running it, so
pick one mechanism and don't run both against the same DuckDB file at
once (the merge step in `update_data.py` isn't designed for two concurrent
writers).

**Cron-style schedulers can silently miss a run.** Confirmed live: a host
sleep/pause (e.g. a Docker Desktop VM suspending on macOS) can cause
`AsyncIOScheduler`'s job to be dropped with zero log output if the miss
exceeds `misfire_grace_time` — already set generously (1 hour) in
`telegram_bot.py`, but that only shrinks the window, it doesn't guarantee
catching an arbitrarily long pause. The real backstop is a startup
freshness check: if the data is older than `UPDATE_STALE_THRESHOLD_HOURS`
(`.env`, default 30) when the bot starts, it runs a catch-up sync
immediately rather than waiting for the next scheduled tick. This is
mostly a non-issue on a real always-on Linux server (nothing to sleep),
but it's what actually recovered the data after the exact scenario above
during testing — worth knowing it's there.

First run ever: `update_data.py` refuses to run (exits with an error
rather than an unbounded crawl from 2015) if `data/prozorro.duckdb`
doesn't exist yet or has no tenders. Run the one-time backfill first:

```bash
python extract_prozorro.py && python transform_prozorro.py && python load_chroma.py
```

## 2. LLM provider

Set in `.env`:

```
LLM_PROVIDER=ollama            # default — local, free, needs `ollama serve`
# or
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini       # optional, this is already the default
```

For `ollama` when Ollama isn't on `localhost` (a sibling Docker container,
or a separate GPU box), also set `OLLAMA_HOST` — read natively by the
`ollama` python package, not by this project's own code:

```
OLLAMA_HOST=http://ollama:11434              # sibling container, Compose's internal DNS
OLLAMA_HOST=http://203.0.113.10:11434        # separate GPU host
```

## 3. Docker

```bash
cp .env.example .env   # fill in TELEGRAM_BOT_TOKEN and your chosen LLM_PROVIDER
```

**Cheap VPS + OpenAI** (no GPU, no local model):

```bash
docker compose up -d
```

**GPU host running Ollama, as a sibling container:**

```bash
# .env: LLM_PROVIDER=ollama, OLLAMA_HOST=http://ollama:11434
docker compose --profile local-llm up -d
docker compose exec ollama ollama pull qwen2.5:14b-instruct   # once
```

**GPU host running Ollama, on a separate machine from the bot:**

```bash
# .env: LLM_PROVIDER=ollama, OLLAMA_HOST=http://<gpu-host-ip>:11434
docker compose up -d   # no need for the local `ollama` service at all
```

`data/` and `chroma_db/` are bind-mounted to the host (see
`docker-compose.yml`) so tender data and the vector index survive a
container restart or rebuild — never store them only inside the
container's own writable layer.

Verify the image builds cleanly before deploying:

```bash
docker build -t prozorro-bot .
```

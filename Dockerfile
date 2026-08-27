# Runs telegram_bot.py. Does NOT bundle Ollama — see docker-compose.yml
# for the two supported deployment modes (cheap VPS + OpenAI, or a
# GPU host running Ollama either alongside this container or remotely).
FROM python:3.13-slim

WORKDIR /app

# Build tools for any package that needs to compile from source on this
# platform (most of this project's deps ship manylinux wheels and won't
# need it, but sentence-transformers' dependency chain occasionally does).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

# data/ (raw JSON + prozorro.duckdb) and chroma_db/ are volumes in
# docker-compose.yml — created here so the container has somewhere to
# write before the first volume mount, but the real persistence is the
# host-mounted volume, not this layer.
RUN mkdir -p data/raw chroma_db

# No .env is copied into the image — secrets come from docker-compose's
# env_file at run time, never baked into a built image layer.

CMD ["python", "telegram_bot.py"]

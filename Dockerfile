# Always-on worker image for the Historical RAG Memory Discord bot.
# The bot holds a persistent Discord gateway connection, so it runs as a
# long-lived process (a "worker"), not a request/response web service.
FROM python:3.12-slim

# Logs stream immediately instead of being buffered (so host log viewers
# show progress in real time).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY . .

# Run as a non-root user.
RUN useradd --create-home --uid 10001 botuser \
    && chown -R botuser:botuser /app
USER botuser

# No port is exposed: this is a worker process. The Discord connection is
# outbound-only, and with TURSO_* configured all state lives in Turso, so
# the container filesystem is disposable.
CMD ["python", "main.py"]

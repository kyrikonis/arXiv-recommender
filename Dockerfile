# The API only: it reads the digests the daily workflow writes and records feedback.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never

# Dependencies before the code, so editing the code doesn't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY config.toml profile.toml ./
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uv", "run", "--frozen", "--no-sync", "uvicorn", "recommender.api:app", "--host", "0.0.0.0", "--port", "8000"]

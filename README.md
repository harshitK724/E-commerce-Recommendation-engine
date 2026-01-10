# E-Commerce Recommendation Server

PostgreSQL + Pandas pipeline, PyTorch BPR matrix factorization, FastAPI serving, plus a **LangGraph** agent and an official **MCP** tool server.

Runtime:

- `pandas` — cleansing, ID maps, splits
- `psycopg` — PostgreSQL `COPY` and SQL transforms
- `torch` — CPU matrix factorization (BPR-MF)
- `fastapi` + `uvicorn` — HTTP API
- `langgraph` — router → tools → ground (chitchat skips tools)
- `mcp` — stdio tool server for agent hosts

No SQLAlchemy, LangChain OpenAI SDK, Pandera, SciPy, or CUDA wheels. Optional LLM calls use stdlib `urllib` when `OPENAI_API_KEY` is set.

## Setup

Python 3.11+. From the repo root (WSL or Linux recommended):

```bash
uv sync --extra dev
docker compose up -d   # optional; ingest falls back to Pandas without it
cp .env.example .env
```

Torch is pinned to the CPU index in `pyproject.toml` so NVIDIA packages are not installed.

## Run

```bash
ecomrec ingest --rows 1200000          # or --csv data/raw/2019-Oct.csv
ecomrec train --epochs 8
ecomrec serve                          # http://127.0.0.1:8000/docs
ecomrec agent --user-id 1
ecomrec mcp
```

Without Docker, add `--pandas-only` on ingest. With Postgres up, ingest uses `COPY` plus SQL filters/aggregates, then writes `products` and `user_events`.

Kaggle CSV columns: `event_time`, `event_type`, `product_id`, `category_id`, `category_code`, `brand`, `price`, `user_id`, `user_session`.

## API

- `GET /health`
- `POST /v1/recommendations` `{user_id, k, exclude_seen}`
- `GET /v1/users/{id}/history`
- `GET /v1/products/{id}`
- `POST /v1/similar-items`

Unknown users get `reason: popularity_fallback`.

MCP tools: `get_recommendations`, `get_similar_products`, `get_user_history`, `get_product`.

## Tests

```bash
pytest -q
```

Uses a few-thousand-row synthetic fixture. No Kaggle or Postgres required. After a full ingest, `artifacts/pipeline_stats.json` is the 1M+ processing proof.

## Layout

- `src/ecomrec/data/` ingest, SQL, Pandas, splits
- `src/ecomrec/models/` BPR-MF, train, eval, `RecommendationService`
- `src/ecomrec/serving/` FastAPI, MCP, catalog stores
- `src/ecomrec/agent/` graph router + grounding

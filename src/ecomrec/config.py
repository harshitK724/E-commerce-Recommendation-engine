from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


@dataclass
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://ecomrec:ecomrec@localhost:5432/ecomrec",
    )
    data_dir: Path = Path(os.getenv("ECOMREC_DATA_DIR", "data"))
    artifacts_dir: Path = Path(os.getenv("ECOMREC_ARTIFACTS_DIR", "artifacts"))
    raw_csv_path: Path | None = Path(p) if (p := os.getenv("ECOMREC_RAW_CSV_PATH")) else None
    sample_rows: int = _env_int("ECOMREC_SAMPLE_ROWS", 1_200_000)
    min_user_events: int = 5
    min_item_events: int = 10
    bot_events_per_hour: int = 200
    embedding_dim: int = _env_int("ECOMREC_EMBEDDING_DIM", 32)
    batch_size: int = 2048
    epochs: int = _env_int("ECOMREC_EPOCHS", 8)
    lr: float = 0.05
    weight_decay: float = 1e-5
    negatives: int = 4
    k: int = 10
    eval_k: int = 10
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


settings = Settings()

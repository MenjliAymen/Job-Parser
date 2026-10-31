import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROVIDER_DEFAULTS = {
    "ollama": {
        "base_url": "http://127.0.0.1:11434",
        "fast_model": "mistral",
        "reasoning_model": "deepseek-r1",
        "max_tokens": "",
        "reasoning_effort": "",
    },
    "openai": {
        "base_url": "https://api.groq.com/openai/v1",
        "fast_model": "openai/gpt-oss-20b",
        "reasoning_model": "openai/gpt-oss-120b",
        # Groq counts requested max_tokens against the free tier's 8000 tokens/minute, so stay well below it.
        "max_tokens": "4096",
        "reasoning_effort": "low",
    },
}


def _sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, never overrides variables already set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_base_url: str
    fast_model: str
    reasoning_model: str
    llm_timeout: float
    llm_retries: int
    llm_retry_delay: float
    data_dir: Path
    jobs_db_url: str
    graph_db_url: str
    graph_export_path: Path
    log_level: str
    llm_api_key: str = field(default="", repr=False)
    llm_max_tokens: Optional[int] = None
    llm_reasoning_effort: str = ""
    llm_cache_size: int = 256
    llm_max_concurrency: int = 4

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        if provider not in PROVIDER_DEFAULTS:
            raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDER_DEFAULTS)}, got {provider!r}")
        defaults = PROVIDER_DEFAULTS[provider]
        base_url = os.getenv("LLM_BASE_URL") or (os.getenv("OLLAMA_URL") if provider == "ollama" else None)
        data_dir = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data")).resolve()
        max_tokens = os.getenv("LLM_MAX_TOKENS", defaults["max_tokens"])
        return cls(
            llm_max_tokens=int(max_tokens) if max_tokens else None,
            llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", defaults["reasoning_effort"]),
            llm_cache_size=int(os.getenv("LLM_CACHE_SIZE", "256")),
            llm_max_concurrency=int(os.getenv("LLM_MAX_CONCURRENCY", "4")),
            llm_provider=provider,
            llm_base_url=(base_url or defaults["base_url"]).rstrip("/"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            fast_model=os.getenv("LLM_MODEL", defaults["fast_model"]),
            reasoning_model=os.getenv("LLM_REASONING_MODEL", defaults["reasoning_model"]),
            llm_timeout=float(os.getenv("LLM_TIMEOUT", "300")),
            llm_retries=int(os.getenv("LLM_RETRIES", "3")),
            llm_retry_delay=float(os.getenv("LLM_RETRY_DELAY", "5")),
            data_dir=data_dir,
            jobs_db_url=os.getenv("JOBS_DB_URL", _sqlite_url(data_dir / "jobs.db")),
            graph_db_url=os.getenv("GRAPH_DB_URL", _sqlite_url(data_dir / "graph.db")),
            graph_export_path=Path(os.getenv("GRAPH_EXPORT_PATH", data_dir / "job_tasks_graph.json")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()

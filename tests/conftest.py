import json
from pathlib import Path
from typing import Any, Callable, List, Union

import pytest

from Job_parser.config import Settings
from Job_parser.llm import LLMClient

Reply = Union[str, dict, list, Exception]


class FakeLLM(LLMClient):
    """LLMClient whose replies come from a script instead of Ollama.

    ``replies`` is either a list consumed in order, or a callable ``prompt -> reply``.
    Dicts/lists are serialized to JSON; exceptions are raised.
    """

    def __init__(self, replies: Union[List[Reply], Callable[[str], Reply]]):
        super().__init__(model="fake", retries=1, retry_delay=0)
        self.replies = replies
        self.prompts: List[str] = []
        self.calls: List[dict] = []

    def generate(self, prompt: str, *, json_mode: bool = False, **options: Any) -> str:
        self.prompts.append(prompt)
        self.calls.append({"json_mode": json_mode, **options})
        reply = self.replies(prompt) if callable(self.replies) else self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_provider="ollama",
        llm_base_url="http://ollama.invalid",
        fast_model="fast",
        reasoning_model="reasoning",
        llm_timeout=1,
        llm_retries=1,
        llm_retry_delay=0,
        data_dir=tmp_path,
        jobs_db_url=f"sqlite:///{(tmp_path / 'jobs.db').as_posix()}",
        graph_db_url=f"sqlite:///{(tmp_path / 'graph.db').as_posix()}",
        graph_export_path=tmp_path / "graph.json",
        log_level="WARNING",
    )

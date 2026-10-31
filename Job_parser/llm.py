"""Single client for every LLM call made by the analyzers."""
import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from typing import Any, List, Optional, Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from Job_parser.config import Settings

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


_TRY_AGAIN_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)", re.IGNORECASE)


def _retry_after(response: requests.Response, default: float) -> float:
    """Seconds to wait after a 429: Retry-After header, else Groq's 'try again in 1m6.5s' text."""
    header = response.headers.get("Retry-After")
    if header:
        try:
            return min(float(header), 60.0)
        except ValueError:
            pass
    match = _TRY_AGAIN_RE.search(response.text)
    if match:
        minutes, value, unit = match.groups()
        seconds = float(value) / (1000 if unit == "ms" else 1) + 60 * int(minutes or 0)
        return min(seconds + 0.5, 60.0)
    return default


class LLMError(Exception):
    """The LLM could not be reached or returned something unusable."""

    def __init__(self, message: str, raw_response: Optional[str] = None):
        super().__init__(message)
        self.raw_response = raw_response


def strip_reasoning(text: str) -> str:
    """Remove <think> blocks emitted by reasoning models such as deepseek-r1."""
    text = _THINK_RE.sub("", text)
    # An unterminated block means everything before the closing tag is reasoning.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str, root: str = "{[") -> Any:
    """Return the first JSON value in ``text`` whose opening char is in ``root``.

    Tolerates reasoning blocks, markdown fences, surrounding prose and trailing commas.
    """
    text = _FENCE_RE.sub("", strip_reasoning(text)).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in root:
            continue
        for candidate in (text[i:], _TRAILING_COMMA_RE.sub(r"\1", text[i:])):
            try:
                value, _ = decoder.raw_decode(candidate)
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("No JSON found in LLM response")


class LLMClient:
    """Talks to either Ollama's native API (``provider="ollama"``) or any OpenAI-compatible
    chat-completions API such as Groq, OpenRouter or OpenAI (``provider="openai"``)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 300,
        retries: int = 3,
        retry_delay: float = 5,
        provider: str = "ollama",
        api_key: str = "",
        max_tokens: Optional[int] = None,
        reasoning_effort: str = "",
        cache_size: int = 0,
        max_concurrency: int = 0,
    ):
        if provider not in ("ollama", "openai"):
            raise ValueError(f"Unknown LLM provider {provider!r}")
        self.model = model
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retries = max(1, retries)
        self.retry_delay = retry_delay
        # Default answer budget; a per-call num_predict overrides it. Reasoning tokens count against it.
        self.max_tokens = max_tokens
        # For reasoning models (e.g. gpt-oss): "low" keeps hidden reasoning from eating the answer budget.
        self.reasoning_effort = reasoning_effort
        # Total seconds to spend waiting on 429s per call, on top of the normal retries.
        self.max_rate_limit_wait = 90.0
        self._json_mode_ok = True
        # LRU of raw answers keyed by the full request; 0 disables it.
        self.cache_size = cache_size
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self._cache_lock = threading.Lock()
        # Caps simultaneous requests to this model so bursts queue here instead of tripping rate limits.
        self._slots = threading.BoundedSemaphore(max_concurrency) if max_concurrency > 0 else None

    @classmethod
    def from_settings(cls, settings: Settings, model: str) -> "LLMClient":
        return cls(
            model=model,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout,
            retries=settings.llm_retries,
            retry_delay=settings.llm_retry_delay,
            provider=settings.llm_provider,
            api_key=settings.llm_api_key,
            max_tokens=settings.llm_max_tokens,
            reasoning_effort=settings.llm_reasoning_effort,
            cache_size=settings.llm_cache_size,
            max_concurrency=settings.llm_max_concurrency,
        )

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _request(self, prompt: str, json_mode: bool, options: dict) -> tuple:
        """Return (url, payload, read) for the configured provider; ``read(body)`` -> (text, truncated)."""
        if self.provider == "ollama":
            options = dict(options)
            if self.max_tokens and "num_predict" not in options:
                options["num_predict"] = self.max_tokens
            payload = {"model": self.model, "prompt": prompt, "stream": False, "options": options}
            if json_mode:
                payload["format"] = "json"
            return (
                f"{self.base_url}/api/generate",
                payload,
                lambda body: (body.get("response", ""), body.get("done_reason") == "length"),
            )

        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        for key in ("temperature", "top_p"):
            if key in options:
                payload[key] = options[key]
        max_tokens = options.get("num_predict", self.max_tokens)
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        def read(body: dict) -> tuple:
            choice = (body.get("choices") or [{}])[0]
            return choice.get("message", {}).get("content") or "", choice.get("finish_reason") == "length"

        return f"{self.base_url}/chat/completions", payload, read

    # ----------------- Response cache -----------------

    def _cache_key(self, prompt: str, json_mode: bool, options: dict) -> str:
        material = json.dumps([self.provider, self.base_url, self.model, prompt, json_mode, options],
                              sort_keys=True, default=str)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def _cache_put(self, key: str, text: str) -> None:
        if self.cache_size <= 0:
            return
        with self._cache_lock:
            self._cache[key] = text
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def _cache_discard(self, key: str) -> None:
        with self._cache_lock:
            self._cache.pop(key, None)

    def _post(self, url: str, payload: dict) -> requests.Response:
        if self._slots is None:
            return requests.post(url, json=payload, headers=self._headers, timeout=self.timeout)
        with self._slots:
            return requests.post(url, json=payload, headers=self._headers, timeout=self.timeout)

    # ----------------- Generation -----------------

    def generate(self, prompt: str, *, json_mode: bool = False, use_cache: bool = True, **options: Any) -> str:
        """Return the raw completion text. Raises LLMError after exhausting retries.

        Identical requests are answered from an in-memory LRU cache; ``use_cache=False`` forces a fresh
        generation (which then replaces the cached answer)."""
        key = self._cache_key(prompt, json_mode, options)
        if use_cache:
            cached = self._cache_get(key)
            if cached is not None:
                logger.debug("LLM %s cache hit", self.model)
                return cached
        text = self._generate_uncached(prompt, json_mode, options)
        self._cache_put(key, text)
        return text

    def _generate_uncached(self, prompt: str, json_mode: bool, options: dict) -> str:
        url, payload, read = self._request(prompt, json_mode and self._json_mode_ok, options)

        last_error = "no attempts made"
        rate_limit_waited = 0.0
        attempt = 0
        while attempt < self.retries:
            attempt += 1
            wait = self.retry_delay * attempt
            try:
                start = time.monotonic()
                response = self._post(url, payload)
                if response.status_code == 429:
                    # Rate limited: transient. Wait as long as the server asks, without spending a retry,
                    # until the total rate-limit wait budget is used up.
                    last_error = f"rate limited: {response.text.strip()[:300]}"
                    wait = _retry_after(response, default=wait)
                    if rate_limit_waited + wait <= self.max_rate_limit_wait:
                        rate_limit_waited += wait
                        attempt -= 1
                elif response.status_code == 400 and "json_validate_failed" in response.text and "response_format" in payload:
                    # Groq validates JSON-mode output server-side and rejects the whole generation when the
                    # model slips. Our own extraction tolerates prose around the JSON, so stop using JSON
                    # mode on this client - every rejected generation still costs tokens.
                    logger.info("%s rejected JSON-mode output from %s; disabling JSON mode", self.provider, self.model)
                    self._json_mode_ok = False
                    payload = {k: v for k, v in payload.items() if k != "response_format"}
                    last_error = "provider rejected the model's JSON"
                    attempt -= 1
                    wait = 0
                elif response.status_code == 400 and "reasoning_effort" in response.text and "reasoning_effort" in payload:
                    # The model is not a reasoning model; drop the option for good.
                    logger.info("%s does not accept reasoning_effort; disabling it", self.model)
                    self.reasoning_effort = ""
                    payload = {k: v for k, v in payload.items() if k != "reasoning_effort"}
                    last_error = "model does not accept reasoning_effort"
                    attempt -= 1
                    wait = 0
                elif 400 <= response.status_code < 500:
                    # Bad request / unknown model / bad key: retrying will not help.
                    raise LLMError(
                        f"{self.provider} rejected the request ({response.status_code}): {response.text.strip()[:500]}"
                    )
                else:
                    response.raise_for_status()
                    text, truncated = read(response.json())
                    text = text.strip()
                    logger.debug("LLM %s answered in %.1fs", self.model, time.monotonic() - start)
                    if truncated:
                        # A cut-off answer is never usable JSON; sampling again often lands under the limit.
                        last_error = (
                            f"answer cut off at the token limit (max_tokens={payload.get('max_tokens', 'default')}); "
                            "raise LLM_MAX_TOKENS or lower LLM_REASONING_EFFORT"
                        )
                    elif text:
                        return text
                    else:
                        last_error = "empty response"
            except LLMError:
                raise
            except (requests.RequestException, ValueError) as e:
                last_error = str(e)
            logger.warning("LLM %s request failed (%d/%d retries left): %s",
                           self.model, self.retries - attempt, self.retries, last_error)
            if attempt < self.retries and wait > 0:
                time.sleep(wait)
        raise LLMError(f"LLM {self.model} unavailable after {self.retries} attempts: {last_error}")

    def list_models(self) -> List[str]:
        """Model ids the provider offers. Raises LLMError when unreachable or unauthorized."""
        url = f"{self.base_url}/api/tags" if self.provider == "ollama" else f"{self.base_url}/models"
        try:
            response = requests.get(url, headers=self._headers, timeout=5)
        except requests.RequestException as e:
            raise LLMError(f"cannot reach {self.base_url}: {e.__class__.__name__}") from e
        if response.status_code >= 400:
            raise LLMError(f"{self.base_url} answered {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            return []
        if self.provider == "ollama":
            return sorted(m.get("name", "") for m in body.get("models", []))
        return sorted(m.get("id", "") for m in body.get("data", []))

    def has_model(self, available: List[str]) -> bool:
        # Ollama reports "mistral:latest" for a model requested as "mistral".
        return any(name == self.model or name.split(":")[0] == self.model for name in available)

    def generate_json(self, prompt: str, *, root: str = "{[", json_mode: bool = False, **options: Any) -> Any:
        raw = self.generate(prompt, json_mode=json_mode, **options)
        try:
            return extract_json(raw, root=root)
        except ValueError as e:
            self._cache_discard(self._cache_key(prompt, json_mode, options))
            raise LLMError(str(e), raw_response=raw) from e

    def generate_model(
        self, prompt: str, model: Type[M], *, parse_retries: int = 1, json_mode: bool = False, **options: Any
    ) -> M:
        """Generate and validate against ``model``. Malformed answers are evicted from the cache and
        regenerated up to ``parse_retries`` times, since sampling the same prompt again usually fixes them."""
        key = self._cache_key(prompt, json_mode, options)
        for attempt in range(parse_retries + 1):
            raw = self.generate(prompt, json_mode=json_mode, use_cache=attempt == 0, **options)
            try:
                return model.model_validate(extract_json(raw, root="{"))
            except (ValueError, ValidationError) as e:
                error = e
                self._cache_discard(key)
                logger.warning("LLM %s answer did not match %s (try %d/%d): %s",
                               self.model, model.__name__, attempt + 1, parse_retries + 1, str(e).splitlines()[0])
        raise LLMError(f"LLM response did not match {model.__name__}: {error}", raw_response=raw) from error

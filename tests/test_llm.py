import pytest
import requests

from Job_parser import llm as llm_module
from Job_parser.llm import LLMClient, LLMError, extract_json
from Job_parser.schemas import JobComparison


@pytest.mark.parametrize("text, expected", [
    ('{"a": 1}', {"a": 1}),
    ('<think>maybe {"a": 0}?</think>\n{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Sure! Here it is: {"a": [1, 2,],}  Hope that helps.', {"a": [1, 2]}),
    ('reasoning without opening tag</think>{"a": 1}', {"a": 1}),
    ('["x", "y"]', ["x", "y"]),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_extract_json_root_skips_leading_list():
    assert extract_json('Scores [1, 2] then {"a": 1}', root="{") == {"a": 1}


def test_extract_json_raises_without_json():
    with pytest.raises(ValueError):
        extract_json("no json here")


class _Response:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text
        self.headers = {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def _client(**kwargs):
    return LLMClient(model="m", base_url="http://x", retry_delay=0, **kwargs)


def test_generate_retries_transient_errors(monkeypatch):
    responses = [requests.ConnectionError("down"), _Response(500), _Response(body={"response": " ok "})]
    sent = []

    def fake_post(url, json, timeout, headers=None):
        sent.append(json)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    assert _client(retries=3).generate("p", json_mode=True, temperature=0.1) == "ok"
    assert len(sent) == 3
    assert sent[0]["format"] == "json"
    assert sent[0]["options"] == {"temperature": 0.1}


def test_generate_does_not_retry_client_errors(monkeypatch):
    calls = []

    def fake_post(url, json, timeout, headers=None):
        calls.append(1)
        return _Response(404, text='{"error":"model \'m\' not found"}')

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    with pytest.raises(LLMError, match="not found"):
        _client(retries=3).generate("p")
    assert len(calls) == 1


def test_openai_provider_request_shape(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout, headers=None):
        sent.update(url=url, json=json, headers=headers)
        return _Response(body={"choices": [{"message": {"content": '{"a": 1}'}}]})

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    client = _client(provider="openai", api_key="secret")
    assert client.generate_json("p", json_mode=True, temperature=0, top_p=1, num_predict=50) == {"a": 1}
    assert sent["url"] == "http://x/chat/completions"
    assert sent["headers"] == {"Authorization": "Bearer secret"}
    assert sent["json"] == {
        "model": "m", "messages": [{"role": "user", "content": "p"}],
        "temperature": 0, "top_p": 1, "max_tokens": 50, "response_format": {"type": "json_object"},
    }


def test_json_validation_failure_retries_without_json_mode(monkeypatch):
    sent = []
    responses = [
        _Response(400, text='{"error":{"code":"json_validate_failed","failed_generation":""}}'),
        _Response(body={"choices": [{"message": {"content": 'Here you go: {"a": 1}'}}]}),
    ]

    def fake_post(url, json, timeout, headers=None):
        sent.append(json)
        return responses.pop(0)

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    client = _client(provider="openai", retries=1)  # the fallback must not spend the only retry
    assert client.generate_json("p", json_mode=True) == {"a": 1}
    assert "response_format" in sent[0] and "response_format" not in sent[1]

    # JSON mode stays off for later calls on the same client: rejected generations cost tokens.
    responses.append(_Response(body={"choices": [{"message": {"content": '{"b": 2}'}}]}))
    assert client.generate_json("p", json_mode=True) == {"b": 2}
    assert "response_format" not in sent[2]


@pytest.mark.parametrize("text, header, expected", [
    ("", "7", 7.0),
    ("Please try again in 6.48s. Need more tokens?", None, 6.98),
    ("Please try again in 1m2.5s.", None, 60.0),
    ("Please try again in 250ms.", None, 0.75),
    ("no hint", None, 3.0),
])
def test_retry_after_parsing(text, header, expected):
    response = _Response(429, text=text)
    if header:
        response.headers = {"Retry-After": header}
    assert llm_module._retry_after(response, default=3.0) == pytest.approx(expected)


def test_rate_limits_do_not_spend_retries(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    limited = [_Response(429, text="Please try again in 1s") for _ in range(4)]
    responses = limited + [_Response(body={"choices": [{"message": {"content": "ok"}}]})]
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: responses.pop(0))
    assert _client(provider="openai", retries=1).generate("p") == "ok"


def test_rate_limit_budget_is_bounded(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: _Response(429, text="Please try again in 59s"))
    client = _client(provider="openai", retries=2)
    with pytest.raises(LLMError, match="rate limited"):
        client.generate("p")


def _choice(content, finish="stop"):
    return _Response(body={"choices": [{"finish_reason": finish, "message": {"content": content}}]})


def test_truncated_answer_is_retried_then_reported(monkeypatch):
    sent = []
    responses = [_choice('{"a": [1, 2', "length"), _choice('{"a": [1, 2]}')]

    def fake_post(url, json, timeout, headers=None):
        sent.append(json)
        return responses.pop(0)

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    client = _client(provider="openai", retries=2, max_tokens=4096, reasoning_effort="low")
    assert client.generate_json("p") == {"a": [1, 2]}
    assert sent[0]["max_tokens"] == 4096 and sent[0]["reasoning_effort"] == "low"

    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: _choice("{", "length"))
    with pytest.raises(LLMError, match="LLM_MAX_TOKENS"):
        client.generate("p")


def test_per_call_num_predict_overrides_default_max_tokens(monkeypatch):
    sent = []
    monkeypatch.setattr(llm_module.requests, "post", lambda url, json, **k: sent.append(json) or _choice("ok"))
    _client(provider="openai", max_tokens=4096).generate("p", num_predict=100)
    assert sent[0]["max_tokens"] == 100


def test_unsupported_reasoning_effort_is_dropped(monkeypatch):
    sent = []
    responses = [_Response(400, text='{"error":{"message":"`reasoning_effort` is not supported with this model"}}'),
                 _choice("ok"), _choice("ok again")]
    monkeypatch.setattr(llm_module.requests, "post", lambda url, json, **k: sent.append(json) or responses.pop(0))
    client = _client(provider="openai", retries=1, reasoning_effort="low")
    assert client.generate("p") == "ok"
    assert client.generate("p") == "ok again"
    assert "reasoning_effort" in sent[0] and "reasoning_effort" not in sent[1] and "reasoning_effort" not in sent[2]


def test_identical_requests_are_served_from_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: calls.append(1) or _choice(f"answer {len(calls)}"))
    client = _client(provider="openai", cache_size=2)

    assert client.generate("p", temperature=0) == "answer 1"
    assert client.generate("p", temperature=0) == "answer 1"            # cache hit
    assert client.generate("p", temperature=0.5) == "answer 2"          # different options, different entry
    assert client.generate("p", temperature=0, use_cache=False) == "answer 3"  # forced refresh replaces entry
    assert client.generate("p", temperature=0) == "answer 3"
    client.generate("q"), client.generate("r")                         # evicts the oldest entries (LRU, size 2)
    assert client.generate("p", temperature=0) == "answer 6"
    assert len(calls) == 6


def test_malformed_answers_are_not_served_from_cache(monkeypatch):
    good = ('{"base_foundation_skills": [], "job1_specific_expertise": [], '
            '"job2_specific_expertise": [], "transferability_score": 50}')
    responses = [_choice("not json"), _choice("still not json"), _choice(good)]
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: responses.pop(0))
    client = _client(provider="openai", cache_size=8)

    with pytest.raises(LLMError):
        client.generate_model("p", JobComparison)
    assert client.generate_model("p", JobComparison).transferability_score == 50


def test_concurrency_limit_caps_parallel_requests(monkeypatch):
    import threading
    import time as real_time

    active, peak, lock = [0], [0], threading.Lock()

    def slow_post(*a, **k):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        real_time.sleep(0.05)
        with lock:
            active[0] -= 1
        return _choice("ok")

    monkeypatch.setattr(llm_module.requests, "post", slow_post)
    client = _client(provider="openai", max_concurrency=2)
    threads = [threading.Thread(target=client.generate, args=(f"p{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 2


def test_rate_limit_is_retried(monkeypatch):
    limited = _Response(429, text="slow down")
    limited.headers = {"Retry-After": "0"}
    responses = [limited, _Response(body={"choices": [{"message": {"content": "ok"}}]})]
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: responses.pop(0))
    assert _client(provider="openai", retries=2).generate("p") == "ok"


def test_generate_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: _Response(body={"response": ""}))
    with pytest.raises(LLMError, match="after 2 attempts"):
        _client(retries=2).generate("p")


def test_generate_model_validation_error_keeps_raw(monkeypatch):
    raw = '{"base_foundation_skills": []}'
    calls = []
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: calls.append(1) or _Response(body={"response": raw}))
    with pytest.raises(LLMError) as exc_info:
        _client().generate_model("p", JobComparison)
    assert exc_info.value.raw_response == raw
    assert len(calls) == 2  # one regeneration before giving up


def test_generate_model_regenerates_malformed_answer(monkeypatch):
    good = ('{"base_foundation_skills": [], "job1_specific_expertise": [], '
            '"job2_specific_expertise": [], "transferability_score": 50}')
    responses = [_Response(body={"response": "Sorry, here is my analysis..."}), _Response(body={"response": good})]
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: responses.pop(0))
    assert _client().generate_model("p", JobComparison).transferability_score == 50

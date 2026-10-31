import pytest
from fastapi.testclient import TestClient

from Job_parser.api.server import app, get_services
from Job_parser.llm import LLMError
from Job_parser.services import build_services
from tests.conftest import FakeLLM


@pytest.fixture
def make_client(settings):
    def make(fast_replies=(), reasoning_replies=()):
        def fake(replies):
            return FakeLLM(replies if callable(replies) else list(replies))

        services = build_services(settings, fake(fast_replies), fake(reasoning_replies))
        app.dependency_overrides[get_services] = lambda: services
        return TestClient(app)
    yield make
    app.dependency_overrides.clear()


def test_root(make_client):
    assert make_client().get("/").json()["status"] == "ok"


def test_compare_jobs(make_client):
    reply = ('<think>{ignored}</think>{"base_foundation_skills": ["Python"], "job1_specific_expertise": [],'
             ' "job2_specific_expertise": ["React"], "transferability_score": "70"}')
    resp = make_client(reasoning_replies=[reply]).post("/compare_jobs", json={"job1": "a", "job2": "b"})

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "comparison": {
        "base_foundation_skills": ["Python"], "job1_specific_expertise": [],
        "job2_specific_expertise": ["React"], "transferability_score": 70.0,
    }}


def test_llm_unavailable_returns_502(make_client):
    client = make_client(reasoning_replies=[LLMError("LLM reasoning unavailable after 3 attempts: down")])
    resp = client.post("/transform_job", json={"description": "Scrape listings"})
    assert resp.status_code == 502
    assert "unavailable" in resp.json()["detail"]


def test_malformed_llm_output_returns_502_with_raw(make_client):
    resp = make_client(fast_replies=["I cannot answer that"] * 2).post("/analyze_intent", json={"description": "x"})
    assert resp.status_code == 502
    assert resp.json()["raw_response"] == "I cannot answer that"


def test_request_validation(make_client):
    client = make_client()
    assert client.post("/graph/ingest_subtasks", json={"subtasks": []}).status_code == 422
    assert client.post("/validate_job_skills", json={"job": "x", "skills": {}, "strictness": 150}).status_code == 422


def test_analyze_jobs_accepts_underscore_id(make_client):
    reply = {"summary": "s", "subtasks": [{"task": "t", "explanation": "e", "group": "Other", "weight": 100}]}
    resp = make_client(fast_replies=[reply]).post("/analyze_jobs", json={
        "text": {"_id": "abc", "title": "T", "summary": "S", "duration": "3 to 6 months"},
    })
    assert resp.status_code == 200
    job = resp.json()["jobs"]
    assert job["job_id"] == 1
    assert job["subtasks"][0]["attributes"]["inherited_time"] == 389.7  # 4.5 months x ~20 h/week


def test_graph_endpoints(make_client):
    client = make_client(fast_replies=lambda prompt: {"match": 0, "score": 0})
    resp = client.post("/graph/ingest_subtasks", json={"subtasks": [{"task": "Write tests", "group": "QA"}]})
    assert resp.json()["ingested"] == 1
    assert client.get("/graph/tasks_by_group", params={"name": "QA"}).json()["count"] == 1
    assert client.get("/export_graph").status_code == 200

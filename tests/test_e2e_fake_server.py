"""End-to-end: API -> real LLMClient over HTTP -> fake LLM server, for both providers."""
import dataclasses
import socket
import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient

from devtools.fake_llm_server import create_app
from Job_parser.api.server import app, get_services
from Job_parser.services import build_services

JOB = (
    "We need a Python developer to scrape Zillow and Redfin listings daily, clean and deduplicate the data, "
    "store it in PostgreSQL and show price trends on a dashboard. Experience with Scrapy and Docker is a plus."
)


@pytest.fixture(scope="module")
def fake_server_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("fake LLM server did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(params=["ollama", "openai"])
def client(request, settings, fake_server_url):
    provider = request.param
    settings = dataclasses.replace(
        settings,
        llm_provider=provider,
        llm_base_url=fake_server_url if provider == "ollama" else f"{fake_server_url}/v1",
        fast_model="mistral",
        reasoning_model="deepseek-r1",
    )
    services = build_services(settings)
    app.dependency_overrides[get_services] = lambda: services
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_reports_reachable_llm(client):
    body = client.get("/health").json()
    assert body["llm"]["reachable"] is True
    assert body["llm"]["models"] == {"fast": "mistral", "reasoning": "deepseek-r1"}


def test_health_flags_unknown_model(client):
    app.dependency_overrides[get_services]().fast_llm.model = "retired-model"
    body = client.get("/health").json()["llm"]
    assert body["reachable"] is False
    assert "retired-model" in body["error"]
    assert "mistral" in body["available_models"] or "mistral:latest" in body["available_models"]


def test_every_endpoint_round_trips(client):
    r = client.post("/parse", json={
        "paragraph": "• Scrape real-estate listings with Scrapy every day\n• Store cleaned records in PostgreSQL",
        "skills": ["Scrapy", "PostgreSQL", "React"],
    })
    assert r.status_code == 200, r.text
    assert [x["tools"] for x in r.json()["result"]] == [["Scrapy"], ["PostgreSQL"]]

    r = client.post("/analyze_jobs", json={"text": {"title": "Zillow scraper", "summary": JOB, "duration": "1 to 3 months, 30+ hrs/week"}})
    assert r.status_code == 200, r.text
    job = r.json()["jobs"]
    assert len(job["subtasks"]) == 4
    assert job["estimated_hours"] == pytest.approx(303.1)
    assert sum(s["attributes"]["inherited_time"] for s in job["subtasks"]) == pytest.approx(303.1, abs=0.5)
    assert [s["attributes"]["remark"] for s in job["subtasks"]] == ["High", "High", "Medium", "Low"]

    r = client.post("/build_profiles", json={"jobs": {"summary": job["summary"], "subtasks": job["subtasks"]}})
    assert r.status_code == 200, r.text
    assert any("Profile:" in line for line in r.json()["profiles"][0]["profile_breakdown"])

    r = client.post("/graph/ingest_subtasks", json={"subtasks": [
        {"task": s["task"], "explanation": s["explanation"], "group": s["group"]} for s in job["subtasks"]
    ]})
    assert r.status_code == 200, r.text
    assert client.get("/graph").json()["nodes"]

    r = client.post("/analyze_intent", json={"description": JOB})
    assert r.status_code == 200, r.text
    assert r.json()["analysis"]["client_technical_expertise"] == "YES"

    # The reasoning model replies with a <think> block first; it must be stripped.
    r = client.post("/transform_job", json={"description": JOB})
    assert r.status_code == 200, r.text
    assert "Web Data Extraction" in r.json()["transformation"]["key_categories"]

    r = client.post("/compare_jobs", json={"job1": JOB, "job2": "Build a React dashboard on PostgreSQL data with Docker."})
    assert r.status_code == 200, r.text
    assert "postgresql" in r.json()["comparison"]["base_foundation_skills"]

    skills = {"Scrapy": "crawling", "PostgreSQL": "storage", "Figma": "design"}
    r = client.post("/validate_job_skills", json={"job": JOB, "skills": skills})
    assert r.status_code == 200, r.text
    validation = r.json()["validation"]
    assert validation["all_skills_required"] is False
    assert validation["unnecessary_skills"] == ["Figma"]
    assert validation["relevant_skills"] == ["Scrapy", "PostgreSQL"]

    r = client.post("/compare_job_skills", json={"job": JOB, "skills": skills})
    assert r.status_code == 200, r.text
    assert r.json()["comparison"]["divergent_skills"] == ["Figma"]

    r = client.post("/compare_skills_lists", json={"list1": skills, "list2": {"scrapy": "x", "Airflow": "y"}})
    assert r.status_code == 200, r.text
    assert r.json()["comparison"]["foundation_skills"] == ["Scrapy"]

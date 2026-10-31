import pytest

from Job_parser.core.freelance_parser import (
    IntelligentFreelanceJobParser, Job, estimate_duration_hours, relative_priority,
)
from Job_parser.llm import LLMError
from tests.conftest import FakeLLM

LLM_REPLY = {
    "summary": "Build a scraper for real-estate listings.",
    "suggested_technologies": ["Scrapy"],
    "tech_domain": "Data Engineering",
    "industry_metadata": ["Real Estate"],
    "subtasks": [
        {"task": "Scrape listings", "explanation": "Crawl Zillow", "group": "backend development", "weight": 70,
         "priority": "high"},
        {"task": "Report", "explanation": "Summarize", "group": "Made-up Group", "weight": "30%"},
    ],
    "assumptions": ["Public data only", "No login"],
}


@pytest.fixture
def make_parser(settings):
    def make(replies):
        return IntelligentFreelanceJobParser(FakeLLM(replies), db_url=settings.jobs_db_url)
    return make


# Every duration string that occurs in the scraped Upwork data (Clustering/sqllite.db).
@pytest.mark.parametrize("duration, hours", [
    ("1 to 3 months, Less than 30 hrs/week", 173.2),
    ("Less than 1 month, Less than 30 hrs/week", 65.0),
    ("1 to 3 months, 30+ hrs/week", 303.1),
    ("1 to 3 months, Hours to be determined", 173.2),
    ("1 to 3 months, Not sure", 173.2),
    ("More than 6 months, 30+ hrs/week", 1364.0),
    ("3 to 6 months, 30+ hrs/week", 682.0),
    ("Less than 1 month, Not sure", 65.0),
    ("More than 6 months, Less than 30 hrs/week", 779.4),
    ("3 to 6 months, Less than 30 hrs/week", 389.7),
])
def test_estimate_duration_hours_on_real_upwork_strings(duration, hours):
    total, basis = estimate_duration_hours(duration)
    assert total == pytest.approx(hours, abs=0.1)
    assert "h/week" in basis


@pytest.mark.parametrize("duration", ["", None, "Hours to be determined", "asap"])
def test_unknown_duration_gives_no_estimate(duration):
    assert estimate_duration_hours(duration) is None


def test_relative_priority_compares_with_even_split():
    assert [relative_priority(s, 4) for s in (0.4, 0.25, 0.2, 0.15)] == ["High", "Medium", "Medium", "Low"]


def test_analyze_with_duration_distributes_hours(make_parser):
    parser = make_parser([LLM_REPLY])
    result = parser.analyze({"title": "Scraper", "summary": "...", "duration": "1 to 3 months, 30+ hrs/week",
                             "skills_list": [" Python  "]})

    assert result.estimated_hours == pytest.approx(303.1)
    assert result.suggested_technologies == ["Python"]  # client skills override the LLM's suggestions
    assert result.assumptions == "Public data only; No login"
    first, second = result.subtasks
    assert first.group == "Backend Development"
    assert second.group == "Other"
    assert first.attributes.inherited_time == pytest.approx(212.2, abs=0.1)
    assert first.attributes.remark == "High"     # from the LLM
    assert second.attributes.remark == "Low"     # no LLM priority: 30% is well under an even 50% split
    assert second.attributes.priority_score == 0.3

    with parser.Session() as session:
        job = session.get(Job, result.job_id)
        assert job.title == "Scraper"
        assert sum(s.inherited_time for s in job.subtasks) == pytest.approx(303.1, abs=0.2)
        assert [t.name for t in job.technologies] == ["Python"]


def test_analyze_without_duration_still_has_effort_and_priority(make_parser):
    result = make_parser([LLM_REPLY]).analyze({"title": "Scraper", "summary": "...", "duration": None})
    assert result.estimated_hours is None
    assert all(s.attributes.inherited_time is None for s in result.subtasks)
    assert [s.attributes.priority_score for s in result.subtasks] == [0.7, 0.3]
    assert result.suggested_technologies == ["Scrapy"]


def test_prompt_does_not_anchor_weights(make_parser):
    llm = FakeLLM([LLM_REPLY])
    IntelligentFreelanceJobParser(llm, db_url="sqlite://").analyze({"title": "t", "summary": "s"})
    assert '"weight": 25' not in llm.prompts[0]
    assert "priority" in llm.prompts[0]


def test_malformed_llm_output_raises(make_parser):
    with pytest.raises(LLMError):
        make_parser(["{\"tech_domain\": \"x\"}"] * 2).analyze({"title": "t", "summary": "s"})

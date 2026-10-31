from Job_parser.core.parser import IntelligentResumeParser
from tests.conftest import FakeLLM

PARAGRAPH = """
Key Responsibilities:
• Build machine learning models for churn prediction
• Deploy models to production using Docker
• ok
"""


def test_extracts_bullets_and_skips_headers():
    parser = IntelligentResumeParser(FakeLLM([]))
    assert parser._extract_responsibility_bullets(PARAGRAPH) == [
        "Build machine learning models for churn prediction",
        "Deploy models to production using Docker",
    ]


def test_analyze_validates_tools_against_skills():
    llm = FakeLLM([{
        "responsibilities": [
            {
                "summary": "Build churn prediction models",
                "tools": ["python ", "Scikit learn", "Kubernetes"],
                "subtasks": [{"task": "Engineer features", "explanation": "From usage logs"}],
            },
            {"summary": "Deploy models to production", "tools": ["docker"], "subtasks": []},
        ]
    }])
    parser = IntelligentResumeParser(llm)

    result = parser.analyze_responsibilities(PARAGRAPH, skills=["Python", "Scikit-learn", "Docker"])

    assert [r.summary for r in result] == ["Build churn prediction models", "Deploy models to production"]
    assert result[0].tools == ["Python", "Scikit-learn"]  # Kubernetes is not in the skills list
    assert len(result[0].tool_confidence) == 2
    assert result[1].tools == ["Docker"]
    assert llm.calls[0]["json_mode"] is True
    assert "Skills List: Python, Scikit-learn, Docker" in llm.prompts[0]


def test_tool_confidence_is_llm_relevance_capped_and_filtered():
    llm = FakeLLM([{
        "responsibilities": [
            {
                "summary": "Lead technical initiatives",
                "tools": [
                    {"name": "python", "relevance": 30},          # below the relevance floor
                    {"name": "SQL", "relevance": 55},
                    {"name": "Power BI", "relevance": 95},
                    {"name": "Airflow", "relevance": 70},
                    {"name": "AWS", "relevance": 65},             # beyond the 3-tool cap
                    {"name": "Kubernetes", "relevance": 99},      # not in the skills list
                ],
                "subtasks": [],
            },
        ]
    }])
    result = IntelligentResumeParser(llm).analyze_responsibilities(
        PARAGRAPH, skills=["Python", "SQL", "Power BI", "Airflow", "AWS"]
    )
    assert result[0].tools == ["Power BI", "Airflow", "AWS"]
    assert result[0].tool_confidence == [95, 70, 65]
    assert "relevance" in llm.prompts[0]


def test_similar_responsibilities_are_merged():
    llm = FakeLLM([{
        "responsibilities": [
            {"summary": "Develop data pipelines", "tools": [], "subtasks": [{"task": "Write ETL jobs"}]},
            {"summary": "Develop data pipeline", "tools": [], "subtasks": [{"task": "Write ETL job"}, {"task": "Schedule DAGs"}]},
        ]
    }])
    result = IntelligentResumeParser(llm).analyze_responsibilities(PARAGRAPH)

    assert len(result) == 1
    assert [s.task for s in result[0].subtasks] == ["Write ETL jobs", "Schedule DAGs"]


def test_no_bullets_skips_llm():
    llm = FakeLLM([])
    assert IntelligentResumeParser(llm).analyze_responsibilities("too short") == []
    assert llm.prompts == []

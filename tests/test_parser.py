from Job_parser.core.parser import IntelligentResumeParser


def test_parser_basic():
    parser = IntelligentResumeParser()
    result = parser.analyze_responsibilities(
        "Build machine learning models and deploy them using Docker."
    )
    assert isinstance(result, list)
    assert "responsibility" in result[0]

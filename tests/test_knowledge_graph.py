import json
import re

from Job_parser.core.knowledge_graph import Node, OllamaGraph
from tests.conftest import FakeLLM


def _graph(settings, replies):
    return OllamaGraph(FakeLLM(replies), db_path=settings.graph_db_url, output_path=str(settings.graph_export_path))


def test_similar_tasks_merge_into_synonyms(settings):
    graph = _graph(settings, lambda prompt: {"match": 1, "score": 85})
    results = graph.process_main_actions([
        {"task": "Android UI Development", "explanation": "", "group": "Mobile Development"},
        {"task": "Build Android user interfaces", "explanation": "", "group": "Mobile Development"},
    ])

    assert len(results) == 1
    node = results[0]
    assert node["name"] == "Android UI Development"
    assert node["synonyms"] == ["Build Android user interfaces"]
    assert node["weight"] == 2
    assert node["groups"] == ["Mobile Development"]


def test_low_score_creates_new_node(settings):
    graph = _graph(settings, lambda prompt: {"match": 1, "score": 40})
    results = graph.process_main_actions([
        {"task": "Write unit tests", "group": "QA"},
        {"task": "Design logo", "group": "QA"},
    ])
    assert len(results) == 2


def test_one_llm_call_per_task_regardless_of_graph_size(settings):
    llm = FakeLLM(lambda prompt: {"match": 0, "score": 0})
    graph = OllamaGraph(llm, db_path=settings.graph_db_url, output_path=str(settings.graph_export_path))
    graph.process_main_actions([{"task": f"Existing task number {i}", "group": "G"} for i in range(40)])
    llm.prompts.clear()

    graph.process_main_actions([{"task": "Brand new task about scraping", "group": "G"}])

    assert len(llm.prompts) == 1
    listed = re.findall(r'^\d+\. "', llm.prompts[0], re.MULTILINE)
    assert len(listed) == graph.shortlist_size


def test_shortlist_prefers_lexically_close_tasks_and_synonyms(settings):
    graph = _graph(settings, lambda prompt: {"match": 0, "score": 0})
    nodes = [Node(id=str(i), name=name, synonyms=syn) for i, (name, syn) in enumerate([
        ("Design logo", []), ("Write unit tests", []), ("Configure CI", ["Set up GitHub Actions pipeline"]),
    ])]
    graph.shortlist_size = 2
    assert [n.name for n in graph._shortlist("Set up a GitHub Actions pipeline", nodes)][0] == "Configure CI"


def test_exact_and_synonym_matches_skip_the_llm(settings):
    llm = FakeLLM(lambda prompt: {"match": 1, "score": 90})
    graph = OllamaGraph(llm, db_path=settings.graph_db_url, output_path=str(settings.graph_export_path))
    graph.process_main_actions([{"task": "Write unit tests", "group": "QA"}, {"task": "Add test coverage", "group": "QA"}])
    calls = len(llm.prompts)
    graph.process_main_actions(["write unit tests", {"task": "add test coverage", "group": "QA"}])
    assert len(llm.prompts) == calls


def test_to_dict_filters_limits_and_exports(settings):
    graph = _graph(settings, lambda prompt: {"match": 0, "score": 0})
    graph.process_main_actions(
        [{"task": f"Scrape site {i}", "group": "Backend Development"} for i in range(5)]
        + [{"task": "Design logo", "group": "UI/UX Design"}]
    )

    full = graph.to_dict()
    assert full["total_tasks"] == 6 and not full["truncated"]
    assert {e["type"] for e in full["edges"]} == {"belongs_to"}

    limited = graph.to_dict(limit_per_group=2)
    assert limited["truncated"]
    assert sum(n["type"] == "task" for n in limited["nodes"]) == 3

    found = graph.to_dict(q="LOGO")
    assert [n["name"] for n in found["nodes"] if n["type"] == "task"] == ["Design logo"]

    exported = json.loads(open(graph.save_to_json(), encoding="utf-8").read())
    assert sum(n["type"] == "task" for n in exported["nodes"]) == 6


def test_tasks_by_group_and_unknown_group(settings):
    graph = _graph(settings, lambda prompt: {"match": 0, "score": 0})
    graph.process_main_actions([{"task": "Write unit tests", "group": "Quality Assurance & Testing"}])
    assert graph.get_tasks_by_group("quality assurance & testing") == [{"name": "Write unit tests", "weight": 1}]
    assert graph.get_tasks_by_group("Nope") == []

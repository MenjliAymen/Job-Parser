"""A stand-in LLM server for running the whole stack without Ollama or an API key.

Speaks both Ollama's native API (/api/generate, /api/tags) and the OpenAI-compatible API
(/v1/chat/completions, /v1/models). It recognises which analyzer sent a prompt and builds a
plausible answer from the prompt's own content using keyword heuristics - deterministic, fast,
and obviously not real intelligence.

    python -m devtools.fake_llm_server --port 11434
    # then: LLM_PROVIDER=ollama (default)                 -> works as-is
    #   or: LLM_PROVIDER=openai LLM_BASE_URL=http://127.0.0.1:11434/v1
"""
import argparse
import json
import re
import asyncio
from typing import Dict, List

from fastapi import FastAPI, Request
from rapidfuzz import fuzz

TECH_TERMS = [
    "python", "javascript", "typescript", "react native", "react", "next.js", "node.js", "django", "flask",
    "fastapi", "sql", "postgresql", "mysql", "mongodb", "aws", "gcp", "azure", "docker", "kubernetes",
    "terraform", "scrapy", "selenium", "playwright", "beautifulsoup", "beautiful soup", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch", "llm", "openai", "langchain", "rag", "power bi", "tableau",
    "excel", "google sheets", "figma", "swift", "kotlin", "flutter", "wordpress", "shopify", "php", "laravel",
    "java", "spring", "golang", "rust", "c#", ".net", "rest api", "api", "graphql", "etl", "airflow", "spark",
    "web scraping", "scraping", "machine learning", "data analysis", "dashboard", "automation", "n8n", "zapier",
    "stripe", "seo", "ios", "android", "ci/cd", "testing", "nlp", "computer vision", "chatbot",
]

GROUP_RULES = [
    (("scrap", "crawl", "extract", "etl", "pipeline", "database", "sql"), "Backend Development"),
    (("react", "frontend", "ui", "interface", "dashboard", "css", "html"), "Frontend Development"),
    (("ios", "android", "mobile", "flutter", "swift", "kotlin"), "Mobile Development"),
    (("model", "machine learning", "ml", "llm", "nlp", "classif", "predict", "analy"), "Data Science & Machine Learning"),
    (("deploy", "docker", "aws", "cloud", "kubernetes", "ci/cd", "server"), "DevOps & Infrastructure"),
    (("test", "qa", "validat", "quality"), "Quality Assurance & Testing"),
    (("design", "figma", "ux"), "UI/UX Design"),
    (("document", "report", "write"), "Business Analysis"),
]

CATEGORY_RULES = [
    (("scrap", "crawl"), "Web Data Extraction"),
    (("clean", "normaliz", "dedup"), "Data Cleaning"),
    (("dashboard", "report", "visuali"), "Data Visualization"),
    (("api", "integrat"), "API Integration"),
    (("automat", "schedul", "n8n", "zapier"), "Workflow Automation"),
    (("database", "sql", "postgres", "mongo"), "Database Management"),
    (("model", "machine learning", "predict", "classif"), "Machine Learning"),
    (("llm", "gpt", "openai", "chatbot", "rag"), "Generative AI"),
    (("react", "frontend", "website", "web app"), "Web Development"),
    (("ios", "android", "mobile"), "Mobile Development"),
    (("analy", "insight", "trend"), "Data Analysis"),
    (("deploy", "cloud", "aws", "docker"), "Cloud Deployment"),
]


def find_terms(text: str) -> List[str]:
    lower = text.lower()
    found = []
    for term in TECH_TERMS:
        if re.search(r"(?<![\w.])" + re.escape(term) + r"(?![\w])", lower) and not any(term in f for f in found):
            found.append(term)
    return found


def first_match(text: str, rules, default: str) -> str:
    lower = text.lower()
    return next((label for keys, label in rules if any(k in lower for k in keys)), default)


def all_matches(text: str, rules) -> List[str]:
    lower = text.lower()
    return [label for keys, label in rules if any(k in lower for k in keys)]


def between(text: str, start: str, end: str) -> str:
    match = re.search(re.escape(start) + r"(.*?)" + (re.escape(end) if end else r"$"), text, re.DOTALL)
    return match.group(1).strip() if match else ""


def sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.split()) >= 3]


def short_phrase(text: str, words: int = 7) -> str:
    return " ".join(re.sub(r"[^\w\s/&+.-]", "", text).split()[:words]).rstrip(".")


def subtasks_for(item: str) -> List[Dict[str, str]]:
    phrase = short_phrase(item, 6).lower()
    return [
        {"task": f"Define requirements for {phrase}", "explanation": "Clarify inputs, outputs and acceptance criteria."},
        {"task": f"Implement {phrase}", "explanation": "Build the core logic with the agreed tools."},
        {"task": f"Validate {phrase}", "explanation": "Test results against sample data and edge cases."},
    ]


# ----------------- One responder per analyzer prompt -----------------

def answer_responsibilities(prompt: str) -> dict:
    bullets = re.findall(r"^\d+\.\s+(.+)$", between(prompt, "information:", "Skills List:"), re.MULTILINE)
    skills = [s.strip() for s in between(prompt, "Skills List:", "\n").split(",") if s.strip() and s.strip() != "None"]
    return {"responsibilities": [
        {
            "summary": short_phrase(b),
            "tools": [
                {"name": s, "relevance": 90 if s.lower() in b.lower() else 60}
                for s in skills if s.lower() in b.lower() or fuzz.partial_ratio(s.lower(), b.lower()) > 90
            ],
            "subtasks": subtasks_for(b),
        }
        for b in bullets
    ]}


_ACTION_RE = re.compile(r"\b(build|develop|create|scrape|pull|design|integrate|deploy|automate|store|show|answer|"
                        r"log|handle|run|clean|train|need|want|looking for)\b", re.IGNORECASE)
_LEAD_RE = re.compile(r"^(?:we|i)\s+(?:need|want|are looking for|would like)\s+(?:a|an|someone|you)?[^,]*?\bto\s+",
                      re.IGNORECASE)


def as_task(sentence: str) -> str:
    """'We need a Python developer to build a scraper that ...' -> 'Build a scraper that ...'"""
    text = _LEAD_RE.sub("", sentence)
    text = re.sub(r"^(?:we|i)\s+(?:need|want|would like)\s+", "Deliver ", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:it|the \w+)\s+(?:should|must|will)\s+", "", text, flags=re.IGNORECASE)
    phrase = short_phrase(text, 9)
    return phrase[:1].upper() + phrase[1:]


def answer_freelance_job(prompt: str) -> dict:
    title = between(prompt, "Title:", "\n")
    summary = between(prompt, "Summary:", "\nSkills:")
    parts = sentences(summary) or [summary or title]
    work = [s for s in parts if _ACTION_RE.search(s) or find_terms(s)] or parts
    steps = (work + [f"Deliver and document {title.lower()}"] * 4)[:4]
    return {
        "summary": f"{title}: {as_task(work[0]).lower()}." if title else work[0][:200],
        "suggested_technologies": find_terms(summary)[:6],
        "tech_domain": first_match(summary, GROUP_RULES, "Other"),
        "industry_metadata": all_matches(summary, CATEGORY_RULES)[:3],
        "subtasks": [
            {"task": as_task(s), "explanation": s, "group": first_match(s, GROUP_RULES, "Other"), "weight": w,
             "priority": p}
            for s, w, p in zip(steps, (40, 30, 20, 10), ("High", "High", "Medium", "Low"))
        ],
        "assumptions": "Generated by the fake LLM server from keyword heuristics.",
    }


def answer_profile(prompt: str) -> str:
    job = between(prompt, "Job:", "\n")
    tasks = re.findall(r"^- (.+?):", between(prompt, "Key Responsibilities:", "Format your response"), re.MULTILINE)
    lines = [f"Job: {job}", f"└── Problem: Deliver {job.lower()}"]
    for i, task in enumerate(tasks):
        last = i == len(tasks) - 1
        branch, pad = ("└──", "    ") if last else ("├──", "│   ")
        terms = find_terms(task) or ["python", "problem solving"]
        lines.append(f"    {branch} Responsibility: {task}")
        lines.append(f"    {pad}└── Profile: {terms[0].title()} Developer")
        for j, term in enumerate(terms[:3]):
            mark = "└──" if j == len(terms[:3]) - 1 else "├──"
            lines.append(f"    {pad}    {mark} Skill: {term}, score: {4 - j} // needed for {task.lower()}")
    return "\n".join(lines)


def answer_task_match(prompt: str) -> dict:
    task = between(prompt, 'New task: "', '"')
    candidates = re.findall(r'^(\d+)\. "(.+)"$', between(prompt, "Existing tasks:", "Two tasks are the same"), re.MULTILINE)
    scored = [(int(fuzz.token_set_ratio(task.lower(), name.lower())), int(i)) for i, name in candidates]
    score, index = max(scored, default=(0, 0))
    return {"match": index if score >= 60 else 0, "score": score}


def answer_intent(prompt: str) -> dict:
    description = between(prompt, "Description:", "Respond with ONLY")
    terms = find_terms(description)
    categories = all_matches(description, CATEGORY_RULES)
    return {
        "client_technical_expertise": "YES" if len(terms) >= 2 else "NO",
        "purpose_analysis": {
            "raw_intent": (sentences(description) or [description])[0][:200],
            "broad_category": categories[0] if categories else "General Software Development",
            "specific_category": categories[1] if len(categories) > 1 else (terms[0].title() if terms else "Custom Development"),
        },
        "confidence_score": round(min(0.95, 0.4 + 0.1 * len(terms)), 2),
    }


def answer_transform(prompt: str) -> dict:
    description = between(prompt, "# JOB DESCRIPTION TO ANALYZE", "# OUTPUT FORMAT")
    categories = all_matches(description, CATEGORY_RULES) or ["Software Development"]
    return {
        "cleaned_job_description": " ".join(sentences(description)[:2]) or description,
        "key_categories": categories[:8],
    }


def answer_compare_jobs(prompt: str) -> dict:
    t1 = set(find_terms(between(prompt, "Job 1 -", "Job 2 -")))
    t2 = set(find_terms(between(prompt, "Job 2 -", "Your task:")))
    union = t1 | t2
    return {
        "base_foundation_skills": sorted(t1 & t2),
        "job1_specific_expertise": sorted(t1 - t2),
        "job2_specific_expertise": sorted(t2 - t1),
        "transferability_score": round(100 * len(t1 & t2) / len(union)) if union else 0,
    }


def _skills_dict(prompt: str, start: str, end: str) -> Dict[str, str]:
    try:
        return json.loads(between(prompt, start, end))
    except json.JSONDecodeError:
        return {}


def _mentioned(skill: str, description: str, text: str) -> bool:
    lower = text.lower()
    return skill.lower() in lower or any(t in lower for t in find_terms(f"{skill} {description}"))


def answer_validate_skills(prompt: str) -> dict:
    job = between(prompt, "Job:", "Skills List:")
    skills = _skills_dict(prompt, "Skills List:", "Strictness Level:")
    unused = [s for s, d in skills.items() if not _mentioned(s, d, job)]
    covered = " ".join(f"{s} {d}" for s, d in skills.items()).lower()
    missing = [t for t in find_terms(job) if t not in covered]
    return {
        "all_skills_required": not unused,
        "relevant_skills": [s for s in skills if s not in unused],
        "unnecessary_skills": unused,
        "missing_skills": missing,
        "explanation": (f"Not needed for this job: {', '.join(unused)}. " if unused else "Every listed skill is used. ")
        + (f"The job also needs: {', '.join(missing)}." if missing else ""),
    }


def answer_job_vs_skills(prompt: str) -> dict:
    job = between(prompt, "Job -", "Skills List:")
    skills = _skills_dict(prompt, "Skills List:", "Your task:")
    covered = " ".join(f"{s} {d}" for s, d in skills.items()).lower()
    return {
        "foundation_skills": [s for s, d in skills.items() if _mentioned(s, d, job)],
        "divergent_skills": [s for s, d in skills.items() if not _mentioned(s, d, job)],
        "missing_skills": [t for t in find_terms(job) if t not in covered],
    }


def answer_skill_lists(prompt: str) -> dict:
    l1 = _skills_dict(prompt, "List 1:", "List 2:")
    l2 = _skills_dict(prompt, "List 2:", "Your task:")
    k1, k2 = {k.lower(): k for k in l1}, {k.lower(): k for k in l2}
    return {
        "foundation_skills": [k1[k] for k in k1 if k in k2],
        "list1_specific_expertise": [k1[k] for k in k1 if k not in k2],
        "list2_specific_expertise": [k2[k] for k in k2 if k not in k1],
    }


# Order matters: the intent prompt also starts with "Analyze this freelance job posting".
ROUTES = [
    ("Client Technical Expertise", answer_intent),
    ("Analyze these job responsibilities", answer_responsibilities),
    ("Analyze this freelance job posting", answer_freelance_job),
    ("Create a detailed technical role profile", answer_profile),
    ("deciding whether a new job task", answer_task_match),
    ("MISSION", answer_transform),
    ("Compare the following two freelance job descriptions", answer_compare_jobs),
    ("against a freelancer's skills list", answer_validate_skills),
    ("with the given skills list", answer_job_vs_skills),
    ("Compare the following two skills lists", answer_skill_lists),
]


def respond(prompt: str, model: str) -> str:
    handler = next((h for marker, h in ROUTES if marker in prompt), None)
    if handler is None:
        return json.dumps({"error": "fake server does not recognise this prompt"})
    result = handler(prompt)
    text = result if isinstance(result, str) else json.dumps(result, indent=2)
    if "r1" in model or "deepseek" in model or "reason" in model:
        # Mimic reasoning models so the client's <think> stripping is exercised.
        text = "<think>\nThe user wants {structured} output. Let me work through it.\n</think>\n" + text
    return text


def create_app(delay: float = 0.0) -> FastAPI:
    app = FastAPI(title="Fake LLM server")
    models = ["mistral", "deepseek-r1", "llama-3.3-70b-versatile"]

    @app.get("/api/tags")
    def tags():
        return {"models": [{"name": m} for m in models]}

    @app.post("/api/generate")
    async def generate(request: Request):
        body = await request.json()
        await asyncio.sleep(delay)
        return {"model": body.get("model"), "response": respond(body.get("prompt", ""), body.get("model", "")), "done": True}

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": m, "object": "model"} for m in models]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        prompt = "\n".join(m.get("content", "") for m in body.get("messages", []) if m.get("role") == "user")
        await asyncio.sleep(delay)
        return {
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": respond(prompt, body.get("model", ""))}}],
        }

    return app


def main():
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--delay", type=float, default=0.3, help="seconds to sleep per call, to feel like a real model")
    args = ap.parse_args()
    uvicorn.run(create_app(args.delay), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

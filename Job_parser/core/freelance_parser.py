import re
import json
import requests
import time
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass

from sqlalchemy import Column, Integer, String, Text, ForeignKey, Float, create_engine
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

Base = declarative_base()
DATABASE_URL = "sqlite:///ollama_graph.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=True)
    summary = Column(Text, nullable=False)
    tech_domain = Column(String, nullable=True)
    assumptions = Column(Text, nullable=True)

    subtasks = relationship("Subtask", back_populates="job", cascade="all, delete-orphan")
    technologies = relationship("Technology", back_populates="job", cascade="all, delete-orphan")


class Subtask(Base):
    __tablename__ = "subtasks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    task = Column(String, nullable=False)
    explanation = Column(Text, nullable=False)
    group = Column(String, nullable=False)
    inherited_time = Column(Float, nullable=True)
    priority_score = Column(Float, nullable=True)
    remark = Column(String, nullable=True)

    job_id = Column(Integer, ForeignKey("jobs.id"))
    job = relationship("Job", back_populates="subtasks")


class Technology(Base):
    __tablename__ = "technologies"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=False)

    job_id = Column(Integer, ForeignKey("jobs.id"))
    job = relationship("Job", back_populates="technologies")


Base.metadata.create_all(engine)


def fix_json_formatting(data: Union[str, Dict, List]) -> Union[Dict, List]:
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            data = data.replace("null", "None").replace("false", "False").replace("true", "True")
            import ast
            try:
                data = ast.literal_eval(data)
            except Exception:
                return data
    if isinstance(data, dict):
        return {k: fix_json_formatting(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [fix_json_formatting(v) for v in data]
    return data

@dataclass
class LLMConfig:
    model: str = "mistral"
    temperature: float = 0.2
    timeout: int = 120
    retry_count: int = 3
    retry_delay: int = 5
    endpoint: str = "http://localhost:11434/api/generate"

ALLOWED_GROUPS = [
    "Frontend Development",
    "Backend Development",
    "Mobile Development",
    "Full-Stack Development",
    "UI/UX Design",
    "DevOps & Infrastructure",
    "Data Science & Machine Learning",
    "Database Management",
    "Quality Assurance & Testing",
    "Cybersecurity",
    "Blockchain & Web3",
    "Game Development",
    "Embedded Systems & IoT",
    "AR/VR Development",
    "Project Management",
    "Product Management",
    "Team Leadership",
    "Business Analysis",
    "Communication Skills",
    "Problem Solving",
    "Research & Development",
    "Other"
]


class IntelligentFreelanceJobParser:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config or LLMConfig()
        self.duration_mappings = {
            "less than 30 hrs/week": 120.0,
            "1 to 3 months": 300.0,
            "3 to 6 months": 600.0,
            "more than 6 months": 1200.0,
        }

    def analyze(self, job: Dict[str, Any]) -> Dict[str, Any]:
        job = fix_json_formatting(job)
        cleaned_skills = self._clean_skills(job.get("skills_list", []))

        duration = job.get("duration", "")
        has_duration = bool(duration and duration.strip())
        duration_hours = self._estimate_duration_hours(duration) if has_duration else 0

        prompt = self._build_prompt(job, cleaned_skills, has_duration, duration_hours)
        response = self._query_llama(prompt)

        if not response:
            return {}

        try:
            data = json.loads(response)

            if cleaned_skills:
                data["suggested_technologies"] = cleaned_skills

            data = self._process_subtasks(data, has_duration, duration_hours)

            with SessionLocal() as session:
                job_id = self._save_to_db(session, job, data)
                data["job_id"] = job_id

            return data
        except json.JSONDecodeError:
            return {}


    def _save_to_db(self, session, raw_job: Dict, parsed: Dict) -> int:
        job_obj = Job(
            title=raw_job.get("title", ""),
            summary=parsed["summary"],
            tech_domain=parsed["tech_domain"],
            assumptions=parsed["assumptions"]
        )
        session.add(job_obj)
        session.flush()

        for tech in parsed.get("suggested_technologies", []):
            session.add(Technology(name=tech, job=job_obj))

        for sub in parsed.get("subtasks", []):
            s = Subtask(
                task=sub["task"],
                explanation=sub["explanation"],
                group=sub.get("group", "Other"),
                inherited_time=sub.get("attributes", {}).get("inherited_time"),
                priority_score=sub.get("attributes", {}).get("priority_score"),
                remark=sub.get("attributes", {}).get("remark"),
                job=job_obj,
            )
            session.add(s)

        session.commit()
        return job_obj.id

    def _clean_skills(self, skills: List[str]) -> List[str]:
        return [re.sub(r"\s+", " ", s.strip()) for s in skills]

    def _process_subtasks(self, data: Dict[str, Any], has_duration: bool, duration_hours: float) -> Dict[str, Any]:
        subtasks = data.get("subtasks", [])
        if has_duration:
            total_weight = sum(t.get("weight", 0) for t in subtasks) or 1
            for task in subtasks:
                weight = task.get("weight", 0) / total_weight
                task["attributes"] = {
                    "inherited_time": round(weight * duration_hours, 1),
                    "priority_score": round(weight, 2),
                    "remark": self._priority_remark(weight),
                }
                task.pop("weight", None)
        else:
            for task in subtasks:
                task.pop("attributes", None)
                task.pop("weight", None)
        return data

    def _priority_remark(self, score: float) -> str:
        if score > 0.6:
            return "High"
        elif score >= 0.3:
            return "Medium"
        return "Low"

    def _estimate_duration_hours(self, duration: str) -> float:
        if not duration:
            return 100.0
        d = duration.lower()
        for k, h in self.duration_mappings.items():
            if k in d:
                return h
        return 100.0

    def _build_prompt(self, job, skills, has_duration, hours) -> str:
        return f"""
Analyze this freelance job posting:

Title: {job.get('title', '')}
Summary: {job.get('summary', '')}

Skills: {', '.join(skills) if skills else 'No explicit skills'}

Provide:
- A concise rewritten summary (max 1 sentence).
- The primary technology domain.
- Industry metadata (or []).
- Exactly 4 subtasks with task + explanation.
- For each subtask, assign **one group ONLY from this fixed list**:
  {", ".join(ALLOWED_GROUPS)}
- If duration exists ({hours}h), distribute weights across subtasks (sum=100).

Return valid JSON:
{{
  "summary": "...",
  "suggested_technologies": ["..."],
  "tech_domain": "...",
  "industry_metadata": [],
  "subtasks": [
    {{"task": "...", "explanation": "...", "group": "...", "weight": 25}}
  ],
  "assumptions": "..."
}}
"""

    def _query_llama(self, prompt: str) -> str:
        for i in range(self.llm_config.retry_count):
            try:
                r = requests.post(
                    self.llm_config.endpoint,
                    json={"model": self.llm_config.model, "prompt": prompt, "stream": False, "format": "json"},
                    timeout=self.llm_config.timeout,
                )
                r.raise_for_status()
                return r.json().get("response", "").strip()
            except Exception as e:
                print(f"[Retry {i+1}] {e}")
                time.sleep(self.llm_config.retry_delay)
        return ""


if __name__ == "__main__":
    parser = IntelligentFreelanceJobParser()
    with open("sample_job.json") as f:
        job_data = json.load(f)
    result = parser.analyze(job_data)
    print(json.dumps(result, indent=2))

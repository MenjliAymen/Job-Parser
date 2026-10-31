import re
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import Column, Integer, String, Text, ForeignKey, Float
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

from Job_parser.db import make_engine
from Job_parser.llm import LLMClient
from Job_parser.schemas import GroupedSubtask, JobAnalysis, RawJobAnalysis, SubtaskAttributes

Base = declarative_base()


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
_GROUP_LOOKUP = {g.lower(): g for g in ALLOWED_GROUPS}
_PRIORITIES = {"high": "High", "medium": "Medium", "low": "Low"}

WEEKS_PER_MONTH = 4.33
# Upwork's project length buckets -> representative number of months.
_MONTH_BUCKETS = [
    (r"less than (?:1|one) month", 0.75, "<1 month"),
    (r"1 to 3 months", 2.0, "1-3 months"),
    (r"3 to 6 months", 4.5, "3-6 months"),
    (r"more than 6 months", 9.0, ">6 months"),
]
# Upwork's weekly-hours buckets -> representative hours per week.
_WEEKLY_BUCKETS = [
    (r"less than 30 hrs?/week", 20.0, "<30 hrs/week"),
    (r"30\+ hrs?/week|more than 30 hrs?/week", 35.0, "30+ hrs/week"),
]
_DEFAULT_WEEKLY_HOURS = 20.0


def estimate_duration_hours(duration: str) -> Optional[Tuple[float, str]]:
    """'1 to 3 months, Less than 30 hrs/week' -> (173.2, '1-3 months (~2 mo) x <30 hrs/week (~20 h)').

    Returns None when the project length is unknown. A missing or undecided weekly load
    ('Hours to be determined', 'Not sure') assumes part-time.
    """
    d = (duration or "").lower()
    months = next(((m, label) for pattern, m, label in _MONTH_BUCKETS if re.search(pattern, d)), None)
    if months is None:
        return None
    weekly = next(((h, label) for pattern, h, label in _WEEKLY_BUCKETS if re.search(pattern, d)), None)
    hours_per_week, weekly_label = weekly or (_DEFAULT_WEEKLY_HOURS, "weekly hours not given")
    total = round(months[0] * WEEKS_PER_MONTH * hours_per_week, 1)
    basis = f"{months[1]} (~{months[0]:g} mo) x {weekly_label} (~{hours_per_week:g} h/week)"
    return total, basis


def relative_priority(share: float, n_subtasks: int) -> str:
    """Fallback when the LLM gives no priority: compare a subtask's effort share with an even split."""
    even = 1 / max(n_subtasks, 1)
    if share >= 1.25 * even:
        return "High"
    if share >= 0.75 * even:
        return "Medium"
    return "Low"


class IntelligentFreelanceJobParser:
    def __init__(self, llm: LLMClient, db_url: str = "sqlite:///jobs.db"):
        self.llm = llm
        self.engine = make_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def analyze(self, job: Dict[str, Any]) -> JobAnalysis:
        cleaned_skills = self._clean_skills(job.get("skills_list") or [])
        estimate = estimate_duration_hours(job.get("duration") or "")

        prompt = self._build_prompt(job, cleaned_skills)
        raw = self.llm.generate_model(prompt, RawJobAnalysis, json_mode=True, temperature=0.2)

        if cleaned_skills:
            raw.suggested_technologies = cleaned_skills
        subtasks = self._process_subtasks(raw, estimate[0] if estimate else None)

        with self.Session() as session:
            job_id = self._save_to_db(session, job.get("title", ""), raw, subtasks)

        return JobAnalysis(
            job_id=job_id,
            summary=raw.summary,
            suggested_technologies=raw.suggested_technologies,
            tech_domain=raw.tech_domain,
            industry_metadata=raw.industry_metadata,
            subtasks=subtasks,
            assumptions=raw.assumptions,
            estimated_hours=estimate[0] if estimate else None,
            estimated_hours_basis=estimate[1] if estimate else None,
        )

    def _save_to_db(self, session, title: str, parsed: RawJobAnalysis, subtasks: List[GroupedSubtask]) -> int:
        job_obj = Job(
            title=title,
            summary=parsed.summary,
            tech_domain=parsed.tech_domain,
            assumptions=parsed.assumptions
        )
        session.add(job_obj)

        for tech in parsed.suggested_technologies:
            session.add(Technology(name=tech, job=job_obj))

        for sub in subtasks:
            attrs = sub.attributes
            session.add(Subtask(
                task=sub.task,
                explanation=sub.explanation,
                group=sub.group,
                inherited_time=attrs.inherited_time if attrs else None,
                priority_score=attrs.priority_score if attrs else None,
                remark=attrs.remark if attrs else None,
                job=job_obj,
            ))

        session.commit()
        return job_obj.id

    def _clean_skills(self, skills: List[str]) -> List[str]:
        return [re.sub(r"\s+", " ", s.strip()) for s in skills if s and s.strip()]

    def _process_subtasks(self, data: RawJobAnalysis, total_hours: Optional[float]) -> List[GroupedSubtask]:
        n = len(data.subtasks)
        weights = [max(t.weight, 0) for t in data.subtasks]
        total_weight = sum(weights)
        result = []
        for task, weight in zip(data.subtasks, weights):
            share = weight / total_weight if total_weight else 1 / max(n, 1)
            result.append(GroupedSubtask(
                task=task.task,
                explanation=task.explanation,
                group=_GROUP_LOOKUP.get(task.group.strip().lower(), "Other"),
                attributes=SubtaskAttributes(
                    inherited_time=round(share * total_hours, 1) if total_hours else None,
                    priority_score=round(share, 2),
                    remark=_PRIORITIES.get(task.priority.strip().lower()) or relative_priority(share, n),
                ),
            ))
        return result

    def _build_prompt(self, job, skills) -> str:
        return f"""
Analyze this freelance job posting:

Title: {job.get('title', '')}
Summary: {job.get('summary', '')}

Skills: {', '.join(skills) if skills else 'No explicit skills'}

Provide:
- A concise rewritten summary (max 1 sentence).
- The primary technology domain.
- Industry metadata (or []).
- Exactly 4 subtasks that together cover the whole job, each with task + explanation.
- For each subtask, assign **one group ONLY from this fixed list**:
  {", ".join(ALLOWED_GROUPS)}
- For each subtask, a "weight": the share of the total work it takes, as an integer. The 4 weights must sum
  to 100 and must reflect real differences in effort (building a scraper is more work than writing a README).
  Only give equal weights if the work is genuinely equal.
- For each subtask, a "priority": "High" if the job fails without it, "Medium" if it is expected but
  secondary, "Low" if it is optional or a nice-to-have.

Return valid JSON:
{{
  "summary": "...",
  "suggested_technologies": ["..."],
  "tech_domain": "...",
  "industry_metadata": [],
  "subtasks": [
    {{"task": "...", "explanation": "...", "group": "...", "weight": <integer>, "priority": "High|Medium|Low"}}
  ],
  "assumptions": "..."
}}
"""

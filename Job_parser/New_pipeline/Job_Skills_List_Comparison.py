import json
from typing import Any, Dict

from Job_parser.llm import LLMClient
from Job_parser.schemas import JobSkillComparison


class JobSkillComparator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def compare(self, job: Dict[str, Any], skills: Dict[str, str]) -> JobSkillComparison:
        return self.llm.generate_model(self._build_prompt(job, skills), JobSkillComparison, temperature=0, top_p=1)

    def _build_prompt(self, job: Dict[str, Any], skills: Dict[str, str]) -> str:
        title = job.get("title", "Job")
        desc = job.get("description", "")

        return f"""
Compare the following freelance job description with the given skills list:

Job - {title}:
{desc}

Skills List:
{json.dumps(skills, indent=2)}

Your task:
1. Identify **Foundation Skills** – overlap between job and skills list.
2. Identify **Divergent Skills** – skills in list but irrelevant to job.
3. Identify **Missing Skills** – job-required skills not in the list.

IMPORTANT:
- Respond ONLY with a valid JSON object.
- Do NOT include explanations, markdown, or commentary.
- JSON must strictly match this format:

{{
  "foundation_skills": ["skill1", "skill2", ...],
  "divergent_skills": ["skill1", "skill2", ...],
  "missing_skills": ["skill1", "skill2", ...]
}}
"""

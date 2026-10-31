from typing import Any, Dict

from Job_parser.llm import LLMClient
from Job_parser.schemas import JobComparison


class JobComparisonAnalyzer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def compare(self, job1: Dict[str, Any], job2: Dict[str, Any]) -> JobComparison:
        return self.llm.generate_model(self._build_prompt(job1, job2), JobComparison, temperature=0, top_p=1)

    def _build_prompt(self, job1: Dict[str, Any], job2: Dict[str, Any]) -> str:
        title1 = job1.get("title", "Job 1")
        desc1 = job1.get("description", "")
        title2 = job2.get("title", "Job 2")
        desc2 = job2.get("description", "")

        return f"""
Compare the following two freelance job descriptions:

Job 1 - {title1}:
{desc1}

Job 2 - {title2}:
{desc2}

Your task:
1. Identify **Base Foundation Skills** (transferable/common skills across both jobs).
2. Identify **Job-Specific Expertise** (unique to each job individually).
3. Provide a **Transferability Score** (0–100) reflecting how transferable skills are between the two roles.

IMPORTANT:
- Respond ONLY with a valid JSON object.
- Do NOT include explanations, markdown, or commentary.
- JSON must strictly match this format:

{{
  "base_foundation_skills": ["skill1", "skill2", ...],
  "job1_specific_expertise": ["skill1", "skill2", ...],
  "job2_specific_expertise": ["skill1", "skill2", ...],
  "transferability_score": 0-100
}}
"""

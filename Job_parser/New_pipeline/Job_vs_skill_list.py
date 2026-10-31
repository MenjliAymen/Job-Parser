import json
from typing import Dict, List

from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

from Job_parser.llm import LLMClient
from Job_parser.schemas import SkillValidation


class JobSkillValidator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def validate(self, job: str, skills: Dict[str, str], strictness: int = 70) -> SkillValidation:
        prompt = self._build_prompt(job, skills, strictness)
        result = self.llm.generate_model(prompt, SkillValidation, temperature=0, top_p=1)
        return self._reconcile(result, list(skills))

    def _reconcile(self, result: SkillValidation, listed: List[str]) -> SkillValidation:
        """Make the answer consistent: every listed skill is either relevant or unnecessary (by its exact
        listed name), missing skills are never ones already listed, and all_skills_required follows."""

        def canonical(name: str):
            match = process.extractOne(name, listed, scorer=fuzz.ratio, processor=default_process)
            return match[0] if match and match[1] >= 85 else None

        relevant = list(dict.fromkeys(c for c in map(canonical, result.relevant_skills) if c))
        unnecessary = [c for c in dict.fromkeys(map(canonical, result.unnecessary_skills)) if c and c not in relevant]
        # Skills the model forgot to classify did not stand out as needed.
        unnecessary += [s for s in listed if s not in relevant and s not in unnecessary]
        missing = [m for m in dict.fromkeys(s.strip() for s in result.missing_skills) if m and not canonical(m)]

        return result.model_copy(update={
            "relevant_skills": relevant,
            "unnecessary_skills": unnecessary,
            "missing_skills": missing,
            "all_skills_required": not unnecessary,
        })

    def _build_prompt(self, job: str, skills: Dict[str, str], strictness: int) -> str:
        return f"""
Compare the following job description against a freelancer's skills list.

Job:
{job}

Skills List:
{json.dumps(skills, indent=2)}

Strictness Level: {strictness} (0 = lenient, 100 = strict). At high strictness a listed skill only counts as
relevant if the job clearly requires it; at low strictness, skills that would plausibly help also count.

Your task - classify the skills:
- relevant_skills: listed skills the job needs (use the exact names from the Skills List)
- unnecessary_skills: listed skills the job does not need (use the exact names from the Skills List)
- missing_skills: skills the job needs that are NOT in the Skills List (short names, e.g. "AWS")
- Every listed skill must appear in exactly one of relevant_skills or unnecessary_skills.
- all_skills_required is true only if unnecessary_skills is empty.
- explanation: 1-3 sentences on the overall fit.

IMPORTANT:
Respond ONLY with valid JSON in this format:

{{
  "all_skills_required": true/false,
  "relevant_skills": ["skill1", ...],
  "unnecessary_skills": ["skill1", ...],
  "missing_skills": ["skill1", ...],
  "explanation": "short explanation"
}}
"""

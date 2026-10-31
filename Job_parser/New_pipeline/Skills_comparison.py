import json
from typing import Dict

from Job_parser.llm import LLMClient
from Job_parser.schemas import SkillsListComparison


class SkillsListComparator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def compare(self, list1: Dict[str, str], list2: Dict[str, str]) -> SkillsListComparison:
        return self.llm.generate_model(self._build_prompt(list1, list2), SkillsListComparison, temperature=0, top_p=1)

    def _build_prompt(self, list1: Dict[str, str], list2: Dict[str, str]) -> str:
        return f"""
Compare the following two skills lists:

List 1:
{json.dumps(list1, indent=2)}

List 2:
{json.dumps(list2, indent=2)}

Your task:
1. Identify **Foundation Skills** (skills present in both lists).
2. Identify **List 1 Specific Expertise** (unique to list 1).
3. Identify **List 2 Specific Expertise** (unique to list 2).

IMPORTANT:
Respond ONLY with a valid JSON object in this format:

{{
  "foundation_skills": ["skill1", "skill2", ...],
  "list1_specific_expertise": ["skill1", "skill2", ...],
  "list2_specific_expertise": ["skill1", "skill2", ...]
}}
"""

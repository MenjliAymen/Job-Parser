import re
import json
import time
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass

# ----------------- CONFIG -----------------
@dataclass
class LLMConfig:
    model: str = "deepseek-r1"
    temperature: float = 0
    top_p: float = 1
    timeout: int = 500
    retry_count: int = 3
    retry_delay: int = 5
    endpoint: str = "http://localhost:11434/api/generate"



class BaseLLMAnalyzer:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config or LLMConfig()

    def _query_llm(self, prompt: str) -> str:

        for i in range(self.llm_config.retry_count):
            try:
                response = requests.post(
                    self.llm_config.endpoint,
                    json={
                        "model": self.llm_config.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.llm_config.temperature,
                            "top_p": self.llm_config.top_p,
                        },
                    },
                    timeout=self.llm_config.timeout,
                )
                response.raise_for_status()
                return response.json().get("response", "").strip()
            except Exception as e:
                print(f"[Retry {i+1}] Error querying LLM: {e}")
                time.sleep(self.llm_config.retry_delay)
        return ""

    def _extract_json(self, response: str) -> Dict[str, Any]:

        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if not json_match:
                return {"error": "No JSON found in response", "raw_response": response}

            raw_json = json_match.group()
            cleaned_json = self._sanitize_json(raw_json)
            return json.loads(cleaned_json)

        except json.JSONDecodeError as e:
            return {"error": f"JSON decode error: {str(e)}", "raw_response": response}

    def _sanitize_json(self, raw: str) -> str:

        cleaned = re.sub(r",\s*}\s*", "}", raw)
        cleaned = re.sub(r",\s*]\s*", "]", cleaned)
        return cleaned.strip()



class JobComparisonAnalyzer(BaseLLMAnalyzer):
    def compare(self, job1: Dict[str, Any], job2: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(job1, job2)
        response = self._query_llm(prompt)
        if not response:
            return {"error": "Failed to get response from LLM"}
        return self._extract_json(response)

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
3. Provide a **Transferability Score** (0–100).

IMPORTANT:
- Respond ONLY with a valid JSON object.
- Do NOT include explanations or commentary.
- JSON must strictly match this format:

{{
  "base_foundation_skills": ["skill1", "skill2", ...],
  "job1_specific_expertise": ["skill1", "skill2", ...],
  "job2_specific_expertise": ["skill1", "skill2", ...],
  "transferability_score": 0-100
}}
"""



class JobSkillValidator(BaseLLMAnalyzer):
    def validate(self, job: str, skills: Dict[str, str], strictness: int = 70) -> Dict[str, Any]:
        prompt = self._build_prompt(job, skills, strictness)
        response = self._query_llm(prompt)
        if not response:
            return {"error": "Failed to get response from LLM"}
        return self._extract_json(response)

    def _build_prompt(self, job: str, skills: Dict[str, str], strictness: int) -> str:
        return f"""
Compare the following job description against the provided skills list.

Job:
{job}

Skills List:
{json.dumps(skills, indent=2)}

Strictness Level: {strictness} (0 = lenient, 100 = strict)

Your task:
- Return whether ALL listed skills are crucial for completing the job.
- If not, list which skills are missing or unnecessary.
- Always include an explanation.

IMPORTANT:
Respond ONLY with valid JSON in this format:

{{
  "all_skills_required": true/false,
  "missing_skills": ["skill1", "skill2", ...],
  "explanation": "short explanation"
}}
"""

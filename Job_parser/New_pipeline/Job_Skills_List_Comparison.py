import re
import json
import requests
import time
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



class JobSkillComparator:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config or LLMConfig()

    def compare(self, job: Dict[str, Any], skills: Dict[str, str]) -> Dict[str, Any]:

        prompt = self._build_prompt(job, skills)
        response = self._query_llm(prompt)

        if not response:
            return {"error": "Failed to get response from LLM"}

        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if not json_match:
                return {"error": "No JSON found in response", "raw_response": response}

            raw_json = json_match.group()
            cleaned_json = self._sanitize_json(raw_json)

            data = json.loads(cleaned_json)
            return data

        except json.JSONDecodeError as e:
            return {"error": f"JSON decode error: {str(e)}", "raw_response": response}

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

    def _sanitize_json(self, raw: str) -> str:

        cleaned = re.sub(r",\s*}", "}", raw)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        return cleaned.strip()

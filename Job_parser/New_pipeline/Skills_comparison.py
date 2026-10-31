import re
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
import requests
import time

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


class SkillsListComparator(BaseLLMAnalyzer):
    def compare(self, list1: Dict[str, str], list2: Dict[str, str]) -> Dict[str, Any]:
        prompt = self._build_prompt(list1, list2)
        response = self._query_llm(prompt)
        if not response:
            return {"error": "Failed to get response from LLM"}
        return self._extract_json(response)

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


if __name__ == "__main__":
    comparator = SkillsListComparator()

    skills1 = {
        "Web scraping": "Collect data from sites",
        "Dashboard design": "UI to visualize data"
    }
    skills2 = {
        "Web scraping": "Collect data from sites",
        "Data cleaning": "Process raw data",
        "Database design": "Organize large datasets"
    }

    result = comparator.compare(skills1, skills2)
    print(json.dumps(result, indent=2))

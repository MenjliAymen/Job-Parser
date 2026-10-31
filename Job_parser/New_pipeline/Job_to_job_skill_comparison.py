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



class JobComparisonAnalyzer:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config or LLMConfig()

    def compare(self, job1: Dict[str, Any], job2: Dict[str, Any]) -> Dict[str, Any]:

        prompt = self._build_prompt(job1, job2)
        response, elapsed = self._query_llm(prompt)

        if not response:
            return {"status": "error", "error": "Failed to get response from LLM", "latency": elapsed}


        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if not json_match:
                return {"status": "error", "error": "No JSON found in response", "raw_response": response, "latency": elapsed}

            raw_json = json_match.group()
            cleaned_json = self._sanitize_json(raw_json)

            data = json.loads(cleaned_json)
            return {"status": "ok", "comparison": data, "latency": elapsed}

        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"JSON decode error: {str(e)}", "raw_response": response, "latency": elapsed}

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

    def _query_llm(self, prompt: str) -> (str, float):
        for i in range(self.llm_config.retry_count):
            try:
                start_time = time.time()
                response = requests.post(
                    self.llm_config.endpoint,
                    json={
                        "model": self.llm_config.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.llm_config.temperature,
                            "top_p": self.llm_config.top_p
                        }
                    },
                    timeout=self.llm_config.timeout,
                )
                elapsed = round(time.time() - start_time, 2)
                print(f"[LLM] Response received in {elapsed:.2f} seconds")

                response.raise_for_status()
                return response.json().get("response", "").strip(), elapsed
            except Exception as e:
                print(f"[Retry {i+1}] Error querying LLM: {e}")
                time.sleep(self.llm_config.retry_delay)
        return "", 0.0

    def _sanitize_json(self, raw: str) -> str:

        cleaned = re.sub(r",\s*}", "}", raw)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        return cleaned.strip()

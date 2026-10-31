import re
import json
import requests
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass



@dataclass
class LLMConfig:
    model: str = "mistral"
    temperature: float = 0.1
    timeout: int = 120
    retry_count: int = 3
    retry_delay: int = 5
    endpoint: str = "http://localhost:11434/api/generate"



class JobIntentAnalyzer:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config or LLMConfig()

    def analyze(self, job: Dict[str, Any]) -> Dict[str, Any]:

        prompt = self._build_prompt(job)
        response = self._query_llama(prompt)

        if not response:
            return {"error": "Failed to get response from LLM"}

        try:

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data
            else:
                return {"error": "No JSON found in response", "raw_response": response}
        except json.JSONDecodeError as e:
            return {"error": f"JSON decode error: {str(e)}", "raw_response": response}

    def _build_prompt(self, job: Dict[str, Any]) -> str:

        title = job.get('title', 'Untitled Job')
        description = job.get('description', 'No description provided')

        return f"""
Analyze this freelance job posting to determine:

1. Client Technical Expertise Assessment:
   - Does the client demonstrate understanding of what's technically required?
   - Are the requirements technically coherent and feasible?
   - Would a technical expert consider this client knowledgeable?
   - Answer with ONLY "YES" or "NO"

2. Job Purpose and Intent Analysis:
   - Raw Intent: What is the core business purpose or goal of this project?
   - Broad Category: What general domain or field does this project belong to?
   - Specific Category: What specific area or specialization within the broad category?

Job Details:
Title: {title}
Description: {description}

Respond with ONLY a JSON object in this exact format:
{{
  "client_technical_expertise": "YES or NO",
  "purpose_analysis": {{
    "raw_intent": "brief description of business purpose",
    "broad_category": "general domain/field",
    "specific_category": "specific area within the broad category"
  }},
  "confidence_score": 0.0-1.0
}}

Provide a confidence score based on how clear the job description is.
"""

    def _query_llama(self, prompt: str) -> str:

        for i in range(self.llm_config.retry_count):
            try:
                response = requests.post(
                    self.llm_config.endpoint,
                    json={
                        "model": self.llm_config.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": self.llm_config.temperature}
                    },
                    timeout=self.llm_config.timeout,
                )
                response.raise_for_status()
                return response.json().get("response", "").strip()
            except Exception as e:
                print(f"[Retry {i + 1}] Error querying LLM: {e}")
                time.sleep(self.llm_config.retry_delay)
        return ""
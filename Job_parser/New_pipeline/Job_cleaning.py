import re
import json
import requests
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass



@dataclass
class LLMConfig:
    model: str = "deepseek-r1"
    temperature: float = 0.2
    timeout: int = 120
    retry_count: int = 3
    retry_delay: int = 5
    endpoint: str = "http://localhost:11434/api/generate"



class JobDescriptionTransformer:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config or LLMConfig()

    def transform(self, job: Dict[str, Any]) -> Dict[str, Any]:

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

        description = job.get('description', 'No description provided')

        return f"""
                 MISSION
                Transform the following freelance job posting into a structured format by generating a cleaned description and general, high-level categories.

                 INSTRUCTIONS

                 1. CLEANED JOB DESCRIPTION
                - Rewrite the description as a single, concise, and objective paragraph.
                - Use active verbs (e.g., "Develop", "Build", "Create", "Analyze").
                - State the goal and primary tasks directly.
                - Remove all meta-commentary (e.g., do not use phrases like "The task involves...").

                 2. KEY CATEGORIES
                - CRITICAL: You MUST describe the NATURE of the work, not the SPECIFIC TOOLS.
                - GENERALIZATION RULE: Translate all specific technologies into their broader field
                - Provide 5-8 general categories that describe the fundamental activities.
                - Always include "Workflow Automation" if the job describes an automated process.

                # JOB DESCRIPTION TO ANALYZE
                {description}

                # OUTPUT FORMAT
                Respond with pure, valid JSON only, using this exact structure:
                {{
                  "cleaned_job_description": "Your concise description here.",
                  "key_categories": ["General Category One", "General Category Two", "General Category Three"]
                }}
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

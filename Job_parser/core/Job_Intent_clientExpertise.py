from typing import Any, Dict

from Job_parser.llm import LLMClient
from Job_parser.schemas import IntentAnalysis


class JobIntentAnalyzer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze(self, job: Dict[str, Any]) -> IntentAnalysis:
        return self.llm.generate_model(self._build_prompt(job), IntentAnalysis, temperature=0.1)

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

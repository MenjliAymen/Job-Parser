from typing import Any, Dict

from Job_parser.llm import LLMClient
from Job_parser.schemas import JobTransformation


class JobDescriptionTransformer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def transform(self, job: Dict[str, Any]) -> JobTransformation:
        return self.llm.generate_model(self._build_prompt(job), JobTransformation, temperature=0.2)

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

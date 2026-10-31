import requests
import time
from typing import List, Dict, Any


class ProfileBuilder:
    def __init__(self, model: str = "mistral", max_retries: int = 3):
        self.model = model
        self.max_retries = max_retries

    def analyze_jobs(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        results = []
        for job in jobs:
            summary = job.get("summary", "No summary")
            subtasks = job.get("subtasks", [])

            if not subtasks:
                results.append({"job_summary": summary, "profile_breakdown": []})
                continue

            prompt = self._build_prompt(summary, subtasks)
            response = self._query_llama(prompt)
            breakdown = response.strip() if response else "[Error] No response"


            breakdown_lines = breakdown.splitlines()

            results.append({
                "job_summary": summary,
                "profile_breakdown": breakdown_lines
            })
        return results

    def _build_prompt(self, summary: str, subtasks: list) -> str:
        task_list = "\n".join(
            f"- {task['task']}: {task['explanation']}"
            for task in subtasks[:8]
        )

        return f"""
Create a detailed technical role profile based on these job requirements:

Job: {summary}
Key Responsibilities:
{task_list}

Format your response EXACTLY as follows using this tree structure:

Job: <summary>
└── Problem: <overall project goal>
    ├── Responsibility: <task1>
    │   └── Profile: <Technology-Specific Role>
    │       ├── Skill: <skill1>, score: <1-5> // <reason>
    │       ├── Skill: <skill2>, score: <1-5> // <reason>
    │       └── Skill: <skill3>, score: <1-5> // <reason>
    ├── Responsibility: <task2>
    │   └── Profile: <Technology-Specific Role>
    │       ├── Skill: <skill4>, score: <1-5> // <reason>
    │       └── Skill: <skill5>, score: <1-5> // <reason>
    └── Responsibility: <task3>
        └── Profile: <Technology-Specific Role>
            ├── Skill: <skill6>, score: <1-5> // <reason>
            ├── Skill: <skill7>, score: <1-5> // <reason>
            └── Skill: <skill8>, score: <1-5> // <reason>

CRITICAL RULES:
1. There is ONLY ONE Problem statement for the entire job
2. Profile names MUST include the primary technology (e.g., "React Developer", "Node.js API Developer")
3. Never use generic role names like "UI Designer" or "Backend Developer"
4. List 2-4 skills per responsibility based on complexity
5. Scores MUST be between 1-5 (whole numbers only)
6. ALWAYS maintain the exact tree structure with proper symbols (└──, ├──, │)
7. Start every responsibility with "├──" or "└──" with correct indentation
"""

    def _query_llama(self, prompt: str) -> str:
        for i in range(self.max_retries):
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.25,
                        "max_tokens": 3500
                    },
                    timeout=200
                )
                return response.json().get("response", "")
            except Exception as e:
                print(f"Attempt {i + 1} failed: {str(e)}")
                time.sleep(5)
        return ""

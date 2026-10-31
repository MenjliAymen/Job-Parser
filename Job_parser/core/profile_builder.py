from typing import Any, Dict, List

from Job_parser.llm import LLMClient
from Job_parser.schemas import Profile


class ProfileBuilder:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze_jobs(self, jobs: List[Dict[str, Any]]) -> List[Profile]:
        results = []
        for job in jobs:
            summary = job.get("summary", "No summary")
            subtasks = job.get("subtasks") or []

            if not subtasks:
                results.append(Profile(job_summary=summary, profile_breakdown=[]))
                continue

            breakdown = self.llm.generate(self._build_prompt(summary, subtasks), temperature=0.25, num_predict=3500)
            results.append(Profile(
                job_summary=summary,
                profile_breakdown=[line for line in breakdown.splitlines() if line.strip()],
            ))
        return results

    def _build_prompt(self, summary: str, subtasks: list) -> str:
        task_list = "\n".join(
            f"- {task.get('task', '')}: {task.get('explanation', '')}"
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

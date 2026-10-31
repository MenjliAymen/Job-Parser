import re
from typing import List, Optional

from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

from Job_parser.llm import LLMClient
from Job_parser.schemas import RawResponsibilities, RawTool, Responsibility, Subtask

MAX_TOOLS = 3
MIN_RELEVANCE = 40

_SKIP_PATTERNS = [
    r"as a .* engineer", r"your responsibilities", r"will include",
    r"position:", r"role:", r"title:", r"duties:", r"^.*:$",
    r"key responsibilities", r"telecommuting", r"multiple positions"
]


class IntelligentResumeParser:
    def __init__(self, llm: LLMClient, skills: Optional[List[str]] = None):
        self.llm = llm
        self.skills = skills if skills else []
        self.min_confidence = 80

    def analyze_responsibilities(self, paragraph: str, skills: Optional[List[str]] = None) -> List[Responsibility]:
        skills = [s.strip() for s in (skills or self.skills) if s.strip()]
        bullets = self._extract_responsibility_bullets(paragraph)
        if not bullets:
            return []

        raw = self.llm.generate_model(
            self._build_prompt(bullets, skills), RawResponsibilities, json_mode=True, temperature=0.1
        )
        refined = [self._refine_responsibility(r.summary, r.tools, r.subtasks, skills) for r in raw.responsibilities]
        return self._merge_similar_responsibilities(refined)

    def _extract_responsibility_bullets(self, paragraph: str) -> List[str]:
        bullets = re.split(r'\n|\r|•|‣|\*|;', paragraph)

        cleaned = []
        for b in bullets:
            clean_b = re.sub(r'^\W+', '', b).strip()
            if len(clean_b.split()) >= 3:
                cleaned.append(clean_b)

        return [
            b for b in cleaned
            if not any(re.search(pattern, b, re.IGNORECASE) for pattern in _SKIP_PATTERNS)
        ]

    def _build_prompt(self, bullets: List[str], skills: List[str]) -> str:
        skills_list = ", ".join(skills) if skills else "None"
        bullets_text = "\n".join([f"{i + 1}. {b}" for i, b in enumerate(bullets)])

        return (
            f"Analyze these job responsibilities and extract structured information:\n"
            f"{bullets_text}\n\n"
            f"Skills List: {skills_list}\n\n"
            f"For EACH responsibility:\n"
            f"1. Summary: A 5–10 word phrase starting with a verb (e.g., 'Develop data pipelines')\n"
            f"2. Tools: the skills from the Skills List a person would actually use to carry out THIS "
            f"responsibility (at most {MAX_TOOLS}, or [] if none), each with a relevance from 0 to 100 "
            f"(100 = impossible without it, 50 = useful but replaceable)\n"
            f"3. Subtasks: 3–5 technical steps with explanations\n\n"
            f"Return JSON format ONLY, as below:\n"
            f"{{\n"
            f"  \"responsibilities\": [\n"
            f"    {{\n"
            f"      \"summary\": \"...\",\n"
            f"      \"tools\": [{{\"name\": \"skill1\", \"relevance\": <0-100>}}],\n"
            f"      \"subtasks\": [\n"
            f"        {{\"task\": \"...\", \"explanation\": \"...\"}},\n"
            f"        ...\n"
            f"      ]\n"
            f"    }}\n"
            f"  ]\n"
            f"}}\n\n"
            f"Rules:\n"
            f"- Use ONLY exact skill names from the Skills List\n"
            f"- Assign a skill ONLY if it is explicitly mentioned or clearly implied (e.g., common use case of the task)\n"
            f"- Leadership, coordination and other soft-skill responsibilities usually need no tools\n"
            f"- Do NOT include tools/technologies in the summary\n"
            f"- Do NOT fabricate or generalize tools not stated/implied\n"
            f"- Subtasks must be technical, specific, and aligned with the responsibility\n"
            f"- Preserve the original order of responsibilities\n"
            f"- OUTPUT JSON ONLY — no extra explanations or text"
        )

    def _refine_responsibility(
        self, summary: str, tools: List[RawTool], subtasks: List[Subtask], skills: List[str]
    ) -> Responsibility:
        # Map each tool onto the canonical skill name (fuzzy, to absorb casing/spelling); its confidence is
        # the LLM's relevance rating. Name similarity says nothing about relevance, so it is only used
        # as a fallback when the model gave no rating.
        best_scores = {}
        for tool in tools:
            if not tool.name or not skills:
                continue
            match = process.extractOne(tool.name, skills, scorer=fuzz.token_sort_ratio, processor=default_process)
            if not match or match[1] < self.min_confidence:
                continue
            canonical, name_score, _ = match
            relevance = max(0.0, min(100.0, tool.relevance)) if tool.relevance is not None else name_score
            if relevance >= MIN_RELEVANCE:
                best_scores[canonical] = max(relevance, best_scores.get(canonical, 0))

        top = sorted(best_scores.items(), key=lambda kv: -kv[1])[:MAX_TOOLS]
        return Responsibility(
            summary=summary,
            tools=[name for name, _ in top],
            tool_confidence=[score for _, score in top],
            subtasks=subtasks,
        )

    def _merge_similar_responsibilities(self, results: List[Responsibility]) -> List[Responsibility]:
        merged = []
        processed_indexes = set()

        for i, result in enumerate(results):
            if i in processed_indexes:
                continue

            similar = [result]
            for j in range(i + 1, len(results)):
                if j not in processed_indexes and fuzz.token_sort_ratio(result.summary, results[j].summary) > 75:
                    similar.append(results[j])
                    processed_indexes.add(j)

            if len(similar) == 1:
                merged.append(result)
                continue

            tool_scores = {}
            merged_subtasks: List[Subtask] = []
            for res in similar:
                for tool, score in zip(res.tools, res.tool_confidence):
                    tool_scores[tool] = max(score, tool_scores.get(tool, 0))
                for subtask in res.subtasks:
                    if not any(fuzz.token_sort_ratio(subtask.task, existing.task) > 85 for existing in merged_subtasks):
                        merged_subtasks.append(subtask)

            top = sorted(tool_scores.items(), key=lambda kv: -kv[1])[:MAX_TOOLS]
            merged.append(Responsibility(
                summary=result.summary,
                tools=[name for name, _ in top],
                tool_confidence=[score for _, score in top],
                subtasks=merged_subtasks,
            ))

        return merged

import re
import json
import requests
import time
from typing import List, Dict, Any, Tuple
from rapidfuzz import fuzz, process


class IntelligentResumeParser:
    def __init__(self, skills: List[str] = None):
        self.retry_count = 3
        self.retry_delay = 5
        self.skills = skills if skills else []
        self.min_confidence = 80

    def analyze_responsibilities(self, paragraph: str) -> List[Dict[str, Any]]:

        bullets = self._extract_responsibility_bullets(paragraph)

        first_pass_results = self._process_bullets_with_llama(bullets)

        final_results = []
        for result in first_pass_results:
            refined = self._refine_responsibility(
                result['summary'],
                result['tools'],
                result['subtasks']
            )
            if refined:
                final_results.append(refined)


        return self._merge_similar_responsibilities(final_results)

    def _extract_responsibility_bullets(self, paragraph: str) -> List[str]:

        bullets = re.split(r'\n|\r|\u2022|\u2023|\*|;', paragraph)


        cleaned = []
        for b in bullets:
            clean_b = re.sub(r'^\W+', '', b).strip()
            if len(clean_b.split()) >= 3:
                cleaned.append(clean_b)


        return [
            b for b in cleaned
            if not any(re.search(pattern, b, re.IGNORECASE) for pattern in [
                r"as a .* engineer", r"your responsibilities", r"will include",
                r"position:", r"role:", r"title:", r"duties:", r"^.*:$",
                r"key responsibilities", r"telecommuting", r"multiple positions"
            ])
        ]

    def _process_bullets_with_llama(self, bullets: List[str]) -> List[Dict[str, Any]]:

        if not bullets:
            return []


        skills_list = ", ".join(self.skills) if self.skills else "None"
        bullets_text = "\n".join([f"{i + 1}. {b}" for i, b in enumerate(bullets)])

        prompt = (
            f"Analyze these job responsibilities and extract structured information:\n"
            f"{bullets_text}\n\n"
            f"Skills List: {skills_list}\n\n"
            f"For EACH responsibility:\n"
            f"1. Summary: A 5–10 word phrase starting with a verb (e.g., 'Develop data pipelines')\n"
            f"2. Tools: List of relevant skills from Skills List (or [] if none)\n"
            f"3. Subtasks: 3–5 technical steps with explanations\n\n"
            f"Return JSON format ONLY, as below:\n"
            f"{{\n"
            f"  \"responsibilities\": [\n"
            f"    {{\n"
            f"      \"summary\": \"...\",\n"
            f"      \"tools\": [\"skill1\", \"skill2\"],\n"
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
            f"- Do NOT include tools/technologies in the summary\n"
            f"- Do NOT fabricate or generalize tools not stated/implied\n"
            f"- Subtasks must be technical, specific, and aligned with the responsibility\n"
            f"- Preserve the original order of responsibilities\n"
            f"- OUTPUT JSON ONLY — no extra explanations or text"
        )


        response = self._query_llama(prompt, timeout=300)
        if not response:
            return []

        try:

            data = json.loads(response)
            return data.get("responsibilities", [])
        except json.JSONDecodeError:

            return []

    def _refine_responsibility(self, summary: str, tools: List[str], subtasks: List[Dict]) -> Dict:

        validated_tools = []
        tool_confidence = []

        for tool in tools:
            if not tool or not self.skills:
                continue
            match = process.extractOne(tool, self.skills, scorer=fuzz.token_sort_ratio)
            if match:
                best_match, score, _ = match
                if score >= self.min_confidence:
                    validated_tools.append(best_match)
                    tool_confidence.append(score)

        validated_tools = list(set(validated_tools))

        valid_subtasks = []
        for subtask in subtasks:
            if isinstance(subtask, dict) and 'task' in subtask and 'explanation' in subtask:
                valid_subtasks.append({
                    "task": subtask['task'],
                    "explanation": subtask['explanation']
                })

        return {
            "summary": summary,
            "tools": validated_tools,
            "tool_confidence": tool_confidence,
            "subtasks": valid_subtasks
        }

    def _merge_similar_responsibilities(self, results: List[Dict]) -> List[Dict]:

        if not results:
            return []

        merged = []

        processed_indexes = set()

        for i, result in enumerate(results):
            if i in processed_indexes:
                continue


            similar = [result]
            for j in range(i + 1, len(results)):
                if j in processed_indexes:
                    continue


                similarity = fuzz.token_sort_ratio(
                    result['summary'],
                    results[j]['summary']
                )

                if similarity > 75:
                    similar.append(results[j])
                    processed_indexes.add(j)


            if len(similar) > 1:
                merged_tools = set()
                merged_subtasks = []

                for res in similar:

                    merged_tools.update(res['tools'])


                    for subtask in res['subtasks']:

                        if not any(
                                fuzz.token_sort_ratio(subtask['task'], existing['task']) > 85
                                for existing in merged_subtasks
                        ):
                            merged_subtasks.append(subtask)

                merged.append({
                    "summary": result['summary'],
                    "tools": list(merged_tools),
                    "subtasks": merged_subtasks
                })
            else:
                merged.append(result)

        return merged

    def _query_llama(self, prompt: str, timeout: int = 120) -> str:

        for i in range(self.retry_count):
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "mistral",
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.1}
                    },
                    timeout=timeout
                )
                response.raise_for_status()
                return response.json().get("response", "").strip()
            except requests.exceptions.Timeout:
                print(f"[Timeout] Retry {i + 1}")
                time.sleep(self.retry_delay * (i + 1))
            except Exception as e:
                print(f"[Error] {str(e)}")
                time.sleep(self.retry_delay)
        return ""
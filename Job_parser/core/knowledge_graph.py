import re
import requests
import json
import os
import uuid
from sqlalchemy import create_engine, Column, String, Integer, Float, ForeignKey, Table, JSON
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.ext.mutable import MutableList

Base = declarative_base()

PREDEFINED_GROUPS = [
    "Frontend Development",
    "Backend Development",
    "Mobile Development",
    "Full-Stack Development",
    "UI/UX Design",
    "DevOps & Infrastructure",
    "Data Science & Machine Learning",
    "Database Management",
    "Quality Assurance & Testing",
    "Cybersecurity",
    "Blockchain & Web3",
    "Game Development",
    "Embedded Systems & IoT",
    "AR/VR Development",
    "Project Management",
    "Product Management",
    "Team Leadership",
    "Business Analysis",
    "Technical Writing",
    "Communication Skills",
    "Problem Solving",
    "Agile Methodology",
    "Sales Engineering",
    "Customer Support",
    "Education & Training",
    "Research & Development"
]



class Group(Base):
    __tablename__ = "groups"
    id = Column(String, primary_key=True)
    name = Column(String, unique=True, index=True)
    description = Column(String, default="")
    nodes = relationship("Node", secondary="group_nodes", back_populates="groups")


class Node(Base):
    __tablename__ = "nodes"
    id = Column(String, primary_key=True)
    name = Column(String, index=True)
    synonyms = Column(MutableList.as_mutable(JSON), default=[])
    weight = Column(Integer, default=1)
    relationships = Column(MutableList.as_mutable(JSON), default=[])
    last_similarity = Column(Float, default=None)

    groups = relationship("Group", secondary="group_nodes", back_populates="nodes")


class GroupNode(Base):
    __tablename__ = "group_nodes"
    group_id = Column(String, ForeignKey("groups.id"), primary_key=True)
    node_id = Column(String, ForeignKey("nodes.id"), primary_key=True)



class OllamaGraph:
    def __init__(self, db_path="sqlite:///job_graph.db", output_path="job_tasks_graph.json", debug=True):

        self.db_path = db_path
        self.output_path = output_path
        self.debug = debug


        self.engine = create_engine(self.db_path, echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)


        self.similarity_cache = {}
        self.similarity_threshold = 60

    def save_to_json(self):
        session = self.Session()
        nodes = []
        edges = []


        group_nodes = {}
        for group in session.query(Group).all():
            group_id = f"group_{group.id}"
            group_nodes[group.id] = group_id


            total_weight = sum(node.weight for node in group.nodes)
            task_count = len(group.nodes)

            nodes.append({
                "id": group_id,
                "label": f"{group.name} ({task_count} tasks, {total_weight} weight)",
                "type": "group",
                "description": group.description,
                "weight": total_weight,
                "task_count": task_count,
                "size": 20 + total_weight * 2
            })

        # Create task nodes
        for node in session.query(Node).all():

            nodes.append({
                "id": node.id,
                "label": f"{node.name} (weight: {node.weight})",
                "type": "task",
                "synonyms": node.synonyms,
                "weight": node.weight,
                "size": 10 + node.weight * 1.5
            })


            for group in node.groups:
                if group.id in group_nodes:
                    edges.append({
                        "source": node.id,
                        "target": group_nodes[group.id],
                        "type": "belongs_to",
                        "weight": 1
                    })


        node_map = {n.id: n for n in session.query(Node).all()}
        for n1 in node_map.values():
            for n2 in node_map.values():
                if n1.id < n2.id:
                    shared_groups = set(g.id for g in n1.groups) & set(g.id for g in n2.groups)
                    if shared_groups:
                        edges.append({
                            "source": n1.id,
                            "target": n2.id,
                            "type": "same_group",
                            "weight": 1
                        })

        graph_data = {"nodes": nodes, "edges": edges}

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)

        session.close()
        print(f"[GRAPH EXPORT] JSON saved to: {os.path.abspath(self.output_path)}")

        return self.output_path


    def _extract_score(self, response_text: str) -> float:
        score_match = re.search(r'Score:\s*(\d{1,3})', response_text, re.IGNORECASE)
        if score_match:
            return float(score_match.group(1))
        numbers = re.findall(r'\b(\d{1,3})\b', response_text)
        for num in numbers:
            value = float(num)
            if 0 <= value <= 100:
                return value
        return 0.0

    def _query_similarity(self, task1: str, task2: str, explanation1: str = "", explanation2: str = "") -> float:
        cache_key = tuple(sorted([task1.lower() + explanation1.lower(), task2.lower() + explanation2.lower()]))
        if cache_key in self.similarity_cache:
            if self.debug:
                print(f"[SIMILARITY CACHE HIT] {task1} vs {task2}: {self.similarity_cache[cache_key]}")
            return self.similarity_cache[cache_key]

        prompt = f"""
        You are a senior technical recruiter evaluating **functional similarity** between two job responsibilities.
        Compare these tasks only on functional equivalence.

        A: "{task1}" ({explanation1})
        B: "{task2}" ({explanation2})

        Respond ONLY in format:
        Score: [0-100]
        """

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt.strip(), "stream": False, "options": {"temperature": 0.0}},
                timeout=60
            )
            response.raise_for_status()
            response_text = response.json().get("response", "").strip()
            score = self._extract_score(response_text)
            score = max(0, min(100, score))
            self.similarity_cache[cache_key] = score

            if self.debug:
                print(f"[SIMILARITY] '{task1}' vs '{task2}': {score}")

            return score
        except Exception as e:
            print(f"[SIMILARITY ERROR] {str(e)}")
            return 0.0


    def _categorize_task(self, task: str, explanation: str = "") -> list:

        groups_list = "\n".join([f"{i + 1}. {group}" for i, group in enumerate(PREDEFINED_GROUPS)])

        prompt = f"""
        You are an expert in categorizing technical tasks and responsibilities into standardized domains.

        TASK TO CATEGORIZE:
        "{task}"

        EXPLANATION (if provided):
        "{explanation}"

        Please categorize this task into ONE OR MORE of the following predefined groups. Choose the most relevant groups:

        {groups_list}

        RESPONSE FORMAT:
        Return a JSON array with the exact group names you've selected. Example:
        ["Frontend Development", "UI/UX Design"]

        ONLY use the group names exactly as listed above. Do not create new groups.
        """

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt.strip(), "stream": False, "options": {"temperature": 0.0}},
                timeout=60
            )
            response.raise_for_status()
            response_text = response.json().get("response", "").strip()


            json_match = re.search(r'\[.*\]', response_text)
            if json_match:
                try:
                    groups = json.loads(json_match.group())

                    filtered_groups = [group for group in groups if group in PREDEFINED_GROUPS]

                    if self.debug:
                        print(f"[CATEGORIZATION] '{task}' -> {filtered_groups}")

                    return filtered_groups
                except json.JSONDecodeError:
                    if self.debug:
                        print(f"[CATEGORIZATION ERROR] Invalid JSON in response: {json_match.group()}")
                    return []
            else:
                print(f"[CATEGORIZATION ERROR] Could not parse response: {response_text}")
                return []

        except Exception as e:
            print(f"[CATEGORIZATION ERROR] {str(e)}")
            return []


    def _get_or_create_group(self, session, group_name):
        if not group_name:
            if self.debug:
                print("[GROUP ERROR] Group name is None or empty")
            return None

        if self.debug:
            print(f"[GROUP LOOKUP] Looking for group: '{group_name}'")

        group = session.query(Group).filter(Group.name.ilike(group_name)).first()
        if group:
            if self.debug:
                print(f"[GROUP FOUND] Found existing group: '{group_name}'")
            return group

        if self.debug:
            print(f"[GROUP CREATE] Creating new group: '{group_name}'")

        new_group = Group(id=str(uuid.uuid4()), name=group_name)
        session.add(new_group)
        session.commit()
        return new_group

    def _add_or_update_main_action(self, session, task, explanation=None, group=None):
        task = task.strip()
        if not task:
            return None

        if self.debug:
            print(f"\n[PROCESSING TASK] '{task}'")
            if group:
                print(f"[PROVIDED GROUP] Restricting search to group: '{group}'")

        group_obj = None
        candidate_nodes = []


        if group:
            group_obj = self._get_or_create_group(session, group)
            if group_obj:
                candidate_nodes = group_obj.nodes
        else:

            candidate_nodes = session.query(Node).all()


        for node in candidate_nodes:
            if node.name.lower() == task.lower():
                if self.debug:
                    print(f"[EXACT STRING MATCH] '{task}' already exists as node name")

                if group_obj and group_obj not in node.groups:
                    node.groups.append(group_obj)
                session.commit()
                return node.id

            if task in node.synonyms:
                if self.debug:
                    print(f"[EXISTING SYNONYM] '{task}' already exists in synonyms of '{node.name}'")

                if group_obj and group_obj not in node.groups:
                    node.groups.append(group_obj)
                session.commit()
                return node.id


        existing = next((n for n in candidate_nodes if n.name.lower() == task.lower()), None)
        if existing:
            if self.debug:
                print(f"[EXACT MATCH FOUND] '{task}' == '{existing.name}'")

            if group_obj and group_obj not in existing.groups:
                existing.groups.append(group_obj)
                if self.debug:
                    print(f"[ADDED TO GROUP] '{existing.name}' added to '{group}'")
            session.commit()
            return existing.id


        best_match, best_score = None, 0.0
        for node in candidate_nodes:
            score = self._query_similarity(task, node.name, explanation or "")
            if score > best_score:
                best_match, best_score = node, score

        if best_match and best_score >= self.similarity_threshold:
            if self.debug:
                print(f"[SIMILARITY MATCH] '{task}' ~ '{best_match.name}' ({best_score})")
            if task not in best_match.synonyms:
                best_match.synonyms.append(task)
                best_match.weight += 1
                if self.debug:
                    print(f"[ADDED SYNONYM] '{task}' added to '{best_match.name}' synonyms")
                    print(f"[WEIGHT INCREMENT] Weight increased to {best_match.weight}")
            else:
                if self.debug:
                    print(f"[SYNONYM EXISTS] '{task}' already in synonyms")
            best_match.last_similarity = best_score
            if group_obj and group_obj not in best_match.groups:
                best_match.groups.append(group_obj)
                if self.debug:
                    print(f"[ADDED TO GROUP] '{best_match.name}' added to '{group}'")
            session.commit()
            return best_match.id


        if self.debug:
            print(f"[NEW NODE] Creating new node for '{task}'")

        new_node = Node(
            id=str(uuid.uuid4()),
            name=task,
            synonyms=[],
            weight=1,
            relationships=[],
            last_similarity=best_score
        )
        if group_obj:
            new_node.groups.append(group_obj)
            if self.debug:
                print(f"[NEW NODE GROUP] '{task}' added to '{group}'")

        session.add(new_node)
        session.commit()
        return new_node.id

    def process_main_actions(self, subtasks):
        if subtasks is None:
            if self.debug:
                print("[WARNING] No subtasks provided (subtasks is None)")
            return []

        session = self.Session()
        processed_node_ids = set()

        if self.debug:
            print(f"\n{'=' * 50}")
            print(f"PROCESSING {len(subtasks)} SUBTASKS")
            print(f"{'=' * 50}")

        # First, process all tasks
        for i, item in enumerate(subtasks):
            if self.debug:
                print(f"\n=== PROCESSING ITEM {i + 1}/{len(subtasks)} ===")

            if isinstance(item, dict):
                task = item.get("task", "")
                explanation = item.get("explanation", "")
                group = item.get("group", None)

                if self.debug:
                    print(f"[TASK DETAILS] Task: '{task}'")
                    if explanation:
                        print(f"[TASK DETAILS] Explanation: '{explanation}'")
                    if group:
                        print(f"[TASK DETAILS] Provided Group: '{group}'")

                node_id = self._add_or_update_main_action(
                    session,
                    task,
                    explanation,
                    group
                )

                if node_id:
                    processed_node_ids.add(node_id)
            else:
                if self.debug:
                    print(f"[TASK DETAILS] Task: '{item}'")

                node_id = self._add_or_update_main_action(session, str(item))
                if node_id:
                    processed_node_ids.add(node_id)

        # After processing all tasks, get the final state of each processed node
        processed_nodes = []
        for node_id in processed_node_ids:
            node = session.query(Node).filter(Node.id == node_id).first()
            if node:
                processed_nodes.append({
                    "id": node.id,
                    "name": node.name,
                    "weight": node.weight,
                    "synonyms": node.synonyms,
                    "groups": [g.name for g in node.groups]
                })

        session.close()
        return processed_nodes


    def get_tasks_by_group(self, group_name):
        session = self.Session()
        group = session.query(Group).filter(Group.name.ilike(group_name)).first()
        if not group:
            return []
        return [{"name": node.name, "weight": node.weight} for node in group.nodes]


def main():

    test_data = {
        "subtasks": [
            {
                "task": "Design Conversion from Figma to iOS (Swift or Objective-C)",
                "explanation": "Converting the provided Figma designs into a functional user interface for an iOS application.",
                "group": "UI/UX Design"
            },
            {
                "task": "Design Conversion from Figma to Android (Kotlin or Java)",
                "explanation": "Converting the provided Figma designs into a functional user interface for an Android application.",
                "group": "UI/UX Design"
            },
            {
                "task": "UI Implementation for iOS Applications",
                "explanation": "Creating user interfaces for iOS apps based on design specifications.",
                "group": "UI/UX Design"
            },
            {
                "task": "Android UI Development",
                "explanation": "Building user interfaces for Android applications from design mockups.",
                "group": "UI/UX Design"
            },
            {
                "task": "Frontend Development using React and TypeScript",
                "explanation": "Developing the frontend of both iOS and Android applications, leveraging React and TypeScript.",
                "group": "Frontend Development"
            },
            {
                "task": "React TypeScript Component Development",
                "explanation": "Creating reusable UI components using React and TypeScript for web applications.",
                "group": "Frontend Development"
            },
            {
                "task": "Quality Assurance and Testing",
                "explanation": "Ensuring the developed application functions as intended and is free of bugs, utilizing unit testing, integration testing, and manual testing.",
                "group": "Quality Assurance & Testing"
            },
            {
                "task": "Software Testing and Quality Control",
                "explanation": "Performing comprehensive testing to ensure software quality and reliability.",
                "group": "Quality Assurance & Testing"
            },
            {
                "task": "Cross-platform Mobile App Development",
                "explanation": "Building applications that work on both iOS and Android platforms using shared codebase.",
                "group": "Mobile Development"
            },
            {
                "task": "Figma to Code Implementation",
                "explanation": "Translating Figma design files into functional code for various platforms.",
                "group": "UI/UX Design"
            },
            {
                "task": "React Native Application Development",
                "explanation": "Creating cross-platform mobile applications using React Native framework.",
                "group": "Mobile Development"
            },
            {
                "task": "Comprehensive Software Testing",
                "explanation": "Implementing various testing methodologies to ensure application quality.",
                "group": "Quality Assurance & Testing"
            }
        ]
    }

    graph = OllamaGraph(debug=True)
    graph.process_main_actions(test_data["subtasks"])
    graph.save_to_json()


if __name__ == "__main__":
    main()
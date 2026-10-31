import json
import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process
from sqlalchemy import Column, String, Integer, Float, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship, selectinload, sessionmaker
from sqlalchemy.ext.mutable import MutableList

from Job_parser.db import make_engine
from Job_parser.llm import LLMClient

logger = logging.getLogger(__name__)

Base = declarative_base()


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


class MatchDecision(BaseModel):
    match: int = Field(0, ge=0, description="1-based index into the candidates, 0 for none")
    score: float = Field(0, ge=0, le=100)


class OllamaGraph:
    """Task knowledge graph. Adding a task costs at most one LLM call, however large the graph is:
    an exact name/synonym match short-circuits, otherwise a lexical shortlist of the closest
    ``shortlist_size`` existing tasks is judged by the LLM in a single prompt."""

    def __init__(self, llm: LLMClient, db_path: str = "sqlite:///graph.db", output_path: str = "job_tasks_graph.json"):
        self.llm = llm
        self.db_path = db_path
        self.output_path = output_path

        self.engine = make_engine(self.db_path)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        self.similarity_threshold = 60
        self.shortlist_size = 8
        # Ingestion reads candidates and then writes; serialising it keeps concurrent requests from
        # creating duplicate nodes for the same task.
        self._write_lock = threading.Lock()

    # ----------------- Reading -----------------

    def save_to_json(self) -> str:
        graph_data = self.to_dict()
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)

        logger.info("Graph exported to %s", os.path.abspath(self.output_path))
        return str(self.output_path)

    def to_dict(self, q: Optional[str] = None, limit_per_group: Optional[int] = None) -> Dict[str, Any]:
        """Group and task nodes plus belongs_to edges. ``q`` filters tasks by name/synonym substring;
        ``limit_per_group`` keeps only each group's heaviest tasks. Tasks shared by groups are implied by
        their belongs_to edges, so no pairwise same_group edges are emitted (they grow quadratically)."""
        needle = (q or "").strip().lower()

        def matches(node: Node) -> bool:
            return not needle or needle in node.name.lower() or any(needle in s.lower() for s in node.synonyms)

        nodes, edges, task_ids = [], [], set()
        truncated = False
        with self.Session() as session:
            groups = session.query(Group).options(selectinload(Group.nodes)).order_by(Group.name).all()
            for group in groups:
                total_weight = sum(node.weight for node in group.nodes)
                kept = sorted((n for n in group.nodes if matches(n)), key=lambda n: (-n.weight, n.name))
                if limit_per_group is not None and len(kept) > limit_per_group:
                    kept, truncated = kept[:limit_per_group], True
                group_id = f"group_{group.id}"
                nodes.append({
                    "id": group_id,
                    "label": f"{group.name} ({len(group.nodes)} tasks, {total_weight} weight)",
                    "name": group.name,
                    "type": "group",
                    "description": group.description,
                    "weight": total_weight,
                    "task_count": len(group.nodes),
                    "size": 20 + total_weight * 2
                })
                for node in kept:
                    if node.id not in task_ids:
                        task_ids.add(node.id)
                        nodes.append(self._task_dict(node))
                    edges.append({"source": node.id, "target": group_id, "type": "belongs_to", "weight": 1})

            # Tasks without any group.
            for node in session.query(Node).filter(~Node.groups.any()).all():
                if matches(node) and node.id not in task_ids:
                    task_ids.add(node.id)
                    nodes.append(self._task_dict(node))

            total_tasks = session.query(Node).count()
        return {"nodes": nodes, "edges": edges, "total_tasks": total_tasks, "truncated": truncated}

    @staticmethod
    def _task_dict(node: Node) -> Dict[str, Any]:
        return {
            "id": node.id,
            "label": f"{node.name} (weight: {node.weight})",
            "name": node.name,
            "type": "task",
            "synonyms": list(node.synonyms),
            "weight": node.weight,
            "size": 10 + node.weight * 1.5
        }

    def get_tasks_by_group(self, group_name: str) -> List[Dict[str, Any]]:
        with self.Session() as session:
            group = session.query(Group).filter(Group.name.ilike(group_name)).first()
            if not group:
                return []
            return [{"name": node.name, "weight": node.weight} for node in group.nodes]

    # ----------------- Matching -----------------

    def _shortlist(self, task: str, candidates: List[Node]) -> List[Node]:
        """The existing tasks lexically closest to ``task``, comparing against names and synonyms."""
        choices: List[Tuple[str, Node]] = [(alias, n) for n in candidates for alias in [n.name, *n.synonyms]]
        if not choices:
            return []
        hits = process.extract(
            task, [alias for alias, _ in choices], scorer=fuzz.token_set_ratio, processor=default_process,
            limit=self.shortlist_size * 3,
        )
        shortlist: List[Node] = []
        for _, _, index in hits:
            node = choices[index][1]
            if node not in shortlist:
                shortlist.append(node)
            if len(shortlist) == self.shortlist_size:
                break
        return shortlist

    def _judge_match(self, task: str, explanation: str, candidates: List[Node]) -> Tuple[Optional[Node], float]:
        """One LLM call: which candidate (if any) is functionally the same task?"""
        listing = "\n".join(f'{i}. "{n.name}"' for i, n in enumerate(candidates, 1))
        prompt = f"""
You are a senior technical recruiter deciding whether a new job task is functionally the same as an existing task.

New task: "{task}" ({explanation})

Existing tasks:
{listing}

Two tasks are the same if a freelancer doing one is doing the same work as the other, even when they are
worded differently or name different tools for the same purpose.

Respond ONLY with JSON: {{"match": <number of the matching existing task, or 0 if none>, "score": <0-100 similarity to that task>}}
"""
        decision = self.llm.generate_model(prompt.strip(), MatchDecision, temperature=0.0)
        if 1 <= decision.match <= len(candidates):
            return candidates[decision.match - 1], decision.score
        return None, decision.score if decision.match == 0 else 0.0

    # ----------------- Writing -----------------

    def _get_or_create_group(self, session, group_name: str) -> Optional[Group]:
        if not group_name:
            return None

        group = session.query(Group).filter(Group.name.ilike(group_name)).first()
        if group:
            return group

        logger.debug("Creating group '%s'", group_name)
        new_group = Group(id=str(uuid.uuid4()), name=group_name)
        session.add(new_group)
        session.commit()
        return new_group

    def _add_or_update_main_action(self, session, task: str, explanation: str = "", group: Optional[str] = None):
        task = task.strip()
        if not task:
            return None

        group_obj = None
        if group:
            group_obj = self._get_or_create_group(session, group)
            candidate_nodes = list(group_obj.nodes) if group_obj else []
        else:
            candidate_nodes = session.query(Node).all()

        def attach(node: Node) -> str:
            if group_obj and group_obj not in node.groups:
                node.groups.append(group_obj)
            session.commit()
            return node.id

        lowered = task.lower()
        for node in candidate_nodes:
            if node.name.lower() == lowered or any(s.lower() == lowered for s in node.synonyms):
                logger.debug("Exact match for '%s' -> '%s'", task, node.name)
                return attach(node)

        best_match, best_score = None, 0.0
        shortlist = self._shortlist(task, candidate_nodes)
        if shortlist:
            best_match, best_score = self._judge_match(task, explanation or "", shortlist)

        if best_match and best_score >= self.similarity_threshold:
            logger.debug("Similarity match '%s' ~ '%s' (%s)", task, best_match.name, best_score)
            if task not in best_match.synonyms:
                best_match.synonyms.append(task)
                best_match.weight += 1
            best_match.last_similarity = best_score
            return attach(best_match)

        logger.debug("New node '%s'", task)
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
        session.add(new_node)
        session.commit()
        return new_node.id

    def process_main_actions(self, subtasks: Optional[List[Any]]) -> List[Dict[str, Any]]:
        if not subtasks:
            return []

        processed_node_ids = []
        with self._write_lock, self.Session() as session:
            for item in subtasks:
                if isinstance(item, dict):
                    node_id = self._add_or_update_main_action(
                        session, item.get("task", ""), item.get("explanation") or "", item.get("group")
                    )
                else:
                    node_id = self._add_or_update_main_action(session, str(item))
                if node_id and node_id not in processed_node_ids:
                    processed_node_ids.append(node_id)

            # Report the final state of each node, after all merges have been applied.
            processed_nodes = []
            for node_id in processed_node_ids:
                node = session.get(Node, node_id)
                if node:
                    processed_nodes.append({
                        "id": node.id,
                        "name": node.name,
                        "weight": node.weight,
                        "synonyms": list(node.synonyms),
                        "groups": [g.name for g in node.groups]
                    })
        return processed_nodes

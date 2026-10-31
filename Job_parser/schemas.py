"""Request, response and LLM-output models shared by the analyzers and the API."""
import re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LenientModel(BaseModel):
    # LLMs routinely add fields we did not ask for; keep them instead of failing.
    model_config = ConfigDict(extra="allow")


# ----------------- Shared pieces -----------------

class Subtask(LenientModel):
    task: str
    explanation: str = ""


class SubtaskAttributes(BaseModel):
    inherited_time: Optional[float] = Field(None, description="Estimated hours; only when the job has a duration")
    priority_score: float = Field(..., description="Share of the total effort, 0-1")
    remark: str = Field(..., description="High / Medium / Low priority")


class GroupedSubtask(Subtask):
    group: str = "Other"
    attributes: Optional[SubtaskAttributes] = None


def _join_if_list(value: Any) -> Any:
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return value


# ----------------- /parse -----------------

class ParseRequest(BaseModel):
    paragraph: str = Field(..., min_length=1)
    skills: List[str] = Field(default_factory=list, description="Skills the LLM may map responsibilities to")


class RawTool(LenientModel):
    name: str
    relevance: Optional[float] = None


class RawResponsibility(LenientModel):
    summary: str
    tools: List[RawTool] = Field(default_factory=list)
    subtasks: List[Subtask] = Field(default_factory=list)

    @field_validator("tools", mode="before")
    @classmethod
    def accept_plain_names(cls, value: Any) -> Any:
        # Older prompts / weaker models return ["Python", ...] instead of [{"name": ..., "relevance": ...}].
        if not isinstance(value, list):
            return value
        return [{"name": v} if isinstance(v, str) else v for v in value if v]


class RawResponsibilities(LenientModel):
    # Required: a missing key means the model ignored the format, which must not look like "no results".
    responsibilities: List[RawResponsibility] = Field(..., min_length=1)


class Responsibility(BaseModel):
    summary: str
    tools: List[str] = Field(default_factory=list)
    tool_confidence: List[float] = Field(
        default_factory=list, description="Per tool: how essential it is to this responsibility, 0-100"
    )
    subtasks: List[Subtask] = Field(default_factory=list)


class ParseResponse(BaseModel):
    result: List[Responsibility]


# ----------------- /analyze_jobs -----------------

class ClientInfo(BaseModel):
    client_from: str
    client_total_spend: Optional[float] = None
    client_hires_stat: Optional[str] = None


class JobActivity(BaseModel):
    Proposals: Optional[str] = None
    Last_viewed_by_client: Optional[str] = None
    Interviewing: Optional[str] = None
    Invites_sent: Optional[str] = None
    Unanswered_invites: Optional[str] = None


class JobPayload(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    title: str
    summary: str
    project_type: Optional[str] = None
    client_info: Optional[ClientInfo] = None
    job_activity_dict: Optional[JobActivity] = None
    skills_list: List[str] = Field(default_factory=list)
    update_status: Optional[bool] = None
    price: Optional[str] = None
    job_type: Optional[str] = None
    duration: Optional[str] = None
    experience: Optional[str] = None
    category: Optional[str] = None
    posted_time: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class JobRequest(BaseModel):
    text: JobPayload


class RawJobSubtask(Subtask):
    group: str = "Other"
    weight: float = 0
    priority: str = ""

    @field_validator("weight", mode="before")
    @classmethod
    def coerce_weight(cls, value: Any) -> Any:
        if isinstance(value, str):
            match = re.search(r"\d+(?:\.\d+)?", value)
            return float(match.group()) if match else 0
        return 0 if value is None else value


class RawJobAnalysis(LenientModel):
    summary: str
    suggested_technologies: List[str] = Field(default_factory=list)
    tech_domain: str = ""
    industry_metadata: List[Any] = Field(default_factory=list)
    subtasks: List[RawJobSubtask] = Field(default_factory=list)
    assumptions: str = ""

    @field_validator("tech_domain", "assumptions", mode="before")
    @classmethod
    def coerce_text(cls, value: Any) -> Any:
        return "" if value is None else _join_if_list(value)


class JobAnalysis(BaseModel):
    job_id: int
    summary: str
    suggested_technologies: List[str]
    tech_domain: str
    industry_metadata: List[Any]
    subtasks: List[GroupedSubtask]
    assumptions: str
    estimated_hours: Optional[float] = Field(None, description="Total hours implied by the job's duration")
    estimated_hours_basis: Optional[str] = Field(None, description="How estimated_hours was derived")


class JobAnalysisResponse(BaseModel):
    jobs: JobAnalysis


# ----------------- /build_profiles -----------------

class ProfileRequest(BaseModel):
    jobs: Union[Dict[str, Any], List[Dict[str, Any]]]


class Profile(BaseModel):
    job_summary: str
    profile_breakdown: List[str]


class ProfileResponse(BaseModel):
    profiles: List[Profile]


# ----------------- /graph -----------------

class SubtaskItem(BaseModel):
    task: str = Field(..., min_length=1)
    explanation: Optional[str] = ""
    group: Optional[str] = None


class IngestSubtasksRequest(BaseModel):
    subtasks: List[SubtaskItem] = Field(..., min_length=1)


class GraphNode(BaseModel):
    id: str
    name: str
    weight: int
    synonyms: List[str]
    groups: List[str]


class IngestSubtasksResponse(BaseModel):
    status: str = "ok"
    ingested: int
    results: List[GraphNode]


class GroupTask(BaseModel):
    name: str
    weight: int


class TasksByGroupResponse(BaseModel):
    group: str
    count: int
    tasks: List[GroupTask]


class ExportGraphResponse(BaseModel):
    message: str


class GraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    total_tasks: int = 0
    truncated: bool = Field(False, description="True when some groups were cut to the per-group limit")


# ----------------- /health -----------------

class LLMStatus(BaseModel):
    provider: str
    base_url: str
    models: Dict[str, str]
    reachable: bool
    error: Optional[str] = None
    available_models: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    llm: LLMStatus


# ----------------- /analyze_intent, /transform_job -----------------

class DescriptionRequest(BaseModel):
    description: str = Field(..., min_length=1, description="Full job description text")


class PurposeAnalysis(LenientModel):
    raw_intent: str
    broad_category: str
    specific_category: str


class IntentAnalysis(LenientModel):
    client_technical_expertise: str
    purpose_analysis: PurposeAnalysis
    confidence_score: float = Field(..., ge=0, le=1)


class IntentAnalysisResponse(BaseModel):
    status: str = "ok"
    analysis: IntentAnalysis


class JobTransformation(LenientModel):
    cleaned_job_description: str
    key_categories: List[str]


class TransformJobResponse(BaseModel):
    status: str = "ok"
    transformation: JobTransformation


# ----------------- Comparisons -----------------

class JobCompareRequest(BaseModel):
    job1: str = Field(..., min_length=1)
    job2: str = Field(..., min_length=1)


class JobComparison(LenientModel):
    base_foundation_skills: List[str]
    job1_specific_expertise: List[str]
    job2_specific_expertise: List[str]
    transferability_score: float = Field(..., ge=0, le=100)


class JobComparisonResponse(BaseModel):
    status: str = "ok"
    comparison: JobComparison


class JobSkillValidationRequest(BaseModel):
    job: str = Field(..., min_length=1, description="Full job description text")
    skills: Dict[str, str] = Field(..., description="Skills list with explanations")
    strictness: int = Field(70, ge=0, le=100, description="Strictness threshold (0-100)")


class SkillValidation(LenientModel):
    all_skills_required: bool
    relevant_skills: List[str] = Field(default_factory=list, description="Listed skills the job needs")
    unnecessary_skills: List[str] = Field(default_factory=list, description="Listed skills the job does not need")
    missing_skills: List[str] = Field(default_factory=list, description="Skills the job needs that are not listed")
    explanation: str = ""


class SkillValidationResponse(BaseModel):
    status: str = "ok"
    validation: SkillValidation


class JobSkillComparisonRequest(BaseModel):
    job: str = Field(..., min_length=1)
    skills: Dict[str, str]


class JobSkillComparison(LenientModel):
    foundation_skills: List[str]
    divergent_skills: List[str]
    missing_skills: List[str]


class JobSkillComparisonResponse(BaseModel):
    status: str = "ok"
    comparison: JobSkillComparison


class SkillsListCompareRequest(BaseModel):
    list1: Dict[str, str] = Field(..., description="First skills list {skill: description}")
    list2: Dict[str, str] = Field(..., description="Second skills list {skill: description}")


class SkillsListComparison(LenientModel):
    foundation_skills: List[str]
    list1_specific_expertise: List[str]
    list2_specific_expertise: List[str]


class SkillsListComparisonResponse(BaseModel):
    status: str = "ok"
    comparison: SkillsListComparison

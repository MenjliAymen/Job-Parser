import logging
import mimetypes
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from Job_parser.config import get_settings
from Job_parser.llm import LLMError
from Job_parser.schemas import (
    DescriptionRequest,
    ExportGraphResponse,
    GraphResponse,
    HealthResponse,
    IngestSubtasksRequest,
    LLMStatus,
    IngestSubtasksResponse,
    IntentAnalysisResponse,
    JobAnalysisResponse,
    JobCompareRequest,
    JobComparisonResponse,
    JobRequest,
    JobSkillComparisonRequest,
    JobSkillComparisonResponse,
    JobSkillValidationRequest,
    ParseRequest,
    ParseResponse,
    ProfileRequest,
    ProfileResponse,
    SkillValidationResponse,
    SkillsListCompareRequest,
    SkillsListComparisonResponse,
    TasksByGroupResponse,
    TransformJobResponse,
)
from Job_parser.services import Services, build_services

logging.basicConfig(level=get_settings().log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
# Windows registries often map .js to text/plain, which strict browsers refuse to execute.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")

app = FastAPI(title="Job Parser API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_services() -> Services:
    return build_services(get_settings())


@app.exception_handler(LLMError)
def llm_error_handler(request: Request, exc: LLMError):
    logger.error("LLM failure on %s: %s", request.url.path, exc)
    content = {"detail": str(exc)}
    if exc.raw_response is not None:
        content["raw_response"] = exc.raw_response
    return JSONResponse(status_code=502, content=content)


app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


@app.get("/")
def root():
    return {"status": "ok", "message": "Job Parser API is running", "ui": "/ui/", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health(services: Services = Depends(get_services)):
    fast, reasoning = services.fast_llm, services.reasoning_llm
    available, error = [], None
    try:
        available = fast.list_models()
    except LLMError as e:
        error = str(e)
    if error is None and available:
        missing = [c.model for c in (fast, reasoning) if not c.has_model(available)]
        if missing:
            error = f"model(s) not available from {fast.provider}: {', '.join(dict.fromkeys(missing))}"
    return HealthResponse(llm=LLMStatus(
        provider=fast.provider,
        base_url=fast.base_url,
        models={"fast": fast.model, "reasoning": reasoning.model},
        reachable=error is None,
        error=error,
        available_models=available,
    ))


@app.get("/graph", response_model=GraphResponse)
def get_graph(
    q: Optional[str] = Query(None, description="Only tasks whose name or synonyms contain this text"),
    limit_per_group: int = Query(100, ge=1, le=10000, description="Heaviest tasks to return per group"),
    services: Services = Depends(get_services),
):
    return services.graph.to_dict(q=q, limit_per_group=limit_per_group)


@app.post("/parse", response_model=ParseResponse)
def parse_responsibility(req: ParseRequest, services: Services = Depends(get_services)):
    return ParseResponse(result=services.resume_parser.analyze_responsibilities(req.paragraph, req.skills))


@app.post("/analyze_jobs", response_model=JobAnalysisResponse)
def analyze_jobs(req: JobRequest, services: Services = Depends(get_services)):
    return JobAnalysisResponse(jobs=services.job_parser.analyze(req.text.model_dump()))


@app.post("/build_profiles", response_model=ProfileResponse)
def build_profiles(req: ProfileRequest, services: Services = Depends(get_services)):
    jobs = req.jobs if isinstance(req.jobs, list) else [req.jobs]
    return ProfileResponse(profiles=services.profile_builder.analyze_jobs(jobs))


@app.post("/graph/ingest_subtasks", response_model=IngestSubtasksResponse)
def ingest_subtasks(payload: IngestSubtasksRequest, services: Services = Depends(get_services)):
    results = services.graph.process_main_actions([item.model_dump() for item in payload.subtasks])
    return IngestSubtasksResponse(ingested=len(results), results=results)


@app.get("/graph/tasks_by_group", response_model=TasksByGroupResponse)
def tasks_by_group(
    name: str = Query(..., min_length=1, description="Exact group name"),
    services: Services = Depends(get_services),
):
    tasks = services.graph.get_tasks_by_group(name)
    return TasksByGroupResponse(group=name, count=len(tasks), tasks=tasks)


@app.get("/export_graph", response_model=ExportGraphResponse)
def export_graph(services: Services = Depends(get_services)):
    path = services.graph.save_to_json()
    return ExportGraphResponse(message=f"Graph exported to {path}")


@app.post("/analyze_intent", response_model=IntentAnalysisResponse)
def analyze_intent(req: DescriptionRequest, services: Services = Depends(get_services)):
    return IntentAnalysisResponse(analysis=services.intent_analyzer.analyze({"description": req.description}))


@app.post("/transform_job", response_model=TransformJobResponse)
def transform_job(req: DescriptionRequest, services: Services = Depends(get_services)):
    return TransformJobResponse(transformation=services.transformer.transform({"description": req.description}))


@app.post("/compare_jobs", response_model=JobComparisonResponse)
def compare_jobs(req: JobCompareRequest, services: Services = Depends(get_services)):
    comparison = services.job_comparator.compare({"description": req.job1}, {"description": req.job2})
    return JobComparisonResponse(comparison=comparison)


@app.post("/validate_job_skills", response_model=SkillValidationResponse)
def validate_job_skills(req: JobSkillValidationRequest, services: Services = Depends(get_services)):
    validation = services.skill_validator.validate(job=req.job, skills=req.skills, strictness=req.strictness)
    return SkillValidationResponse(validation=validation)


@app.post("/compare_job_skills", response_model=JobSkillComparisonResponse)
def compare_job_skills(req: JobSkillComparisonRequest, services: Services = Depends(get_services)):
    comparison = services.job_skill_comparator.compare({"description": req.job}, req.skills)
    return JobSkillComparisonResponse(comparison=comparison)


@app.post("/compare_skills_lists", response_model=SkillsListComparisonResponse)
def compare_skills_lists(req: SkillsListCompareRequest, services: Services = Depends(get_services)):
    return SkillsListComparisonResponse(comparison=services.skills_list_comparator.compare(req.list1, req.list2))

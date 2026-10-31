from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Dict, Any
from fastapi.middleware.cors import CORSMiddleware


from Job_parser.core.parser import IntelligentResumeParser
from Job_parser.core.freelance_parser import IntelligentFreelanceJobParser
from Job_parser.core.profile_builder import ProfileBuilder
from Job_parser.core.knowledge_graph import OllamaGraph
from Job_parser.core.Job_Intent_clientExpertise import JobIntentAnalyzer
from Job_parser.New_pipeline.Job_cleaning import JobDescriptionTransformer
from Job_parser.New_pipeline.Job_to_job_skill_comparison import JobComparisonAnalyzer
from Job_parser.New_pipeline.Job_vs_skill_list import JobSkillValidator
from Job_parser.New_pipeline.Job_Skills_List_Comparison import JobSkillComparator
from Job_parser.New_pipeline.Skills_comparison import SkillsListComparator

app = FastAPI(title="Job Parser API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class ResumeRequest(BaseModel):
    paragraph: str = Field(..., min_length=1)


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
    _id: str
    title: str
    summary: str
    project_type: Optional[str] = None
    client_info: Optional[ClientInfo] = None
    job_activity_dict: Optional[JobActivity] = None
    skills_list: Optional[List[str]] = []
    update_status: Optional[bool] = None
    price: Optional[str] = None
    job_type: Optional[str] = None
    duration: Optional[str] = None
    experience: Optional[str] = None
    category: Optional[str] = None
    posted_time: Optional[str] = None


class JobRequest(BaseModel):
    text: JobPayload


class ProfileRequest(BaseModel):
    jobs: Union[Dict[str, Any], List[Dict[str, Any]]]


class SubtaskItem(BaseModel):
    task: str = Field(..., min_length=1)
    explanation: Optional[str] = ""
    group: Optional[str] = None


class IngestSubtasksRequest(BaseModel):
    subtasks: List[SubtaskItem]

    @validator("subtasks")
    def must_have_items(cls, v):
        if not v:
            raise ValueError("subtasks must contain at least one item")
        return v


class SuggestGroupRequest(BaseModel):
    task: str = Field(..., min_length=1)
    explanation: Optional[str] = ""



class IntentAnalysisRequest(BaseModel):
    description: str = Field(..., min_length=1, description="Full job description text")
    #title: [str] = Field(None, description=" job title")

class JobCompareRequest(BaseModel):
    job1: str
    job2: str

class JobSkillValidationRequest(BaseModel):
    job: str = Field(..., description="Full job description text")
    skills: Dict[str, str] = Field(..., description="Skills list with explanations")
    strictness: Optional[int] = Field(70, ge=0, le=100, description="Strictness threshold (0–100)")

class JobSkillComparisonRequest(BaseModel):  # NEW
    job: str
    skills: Dict[str, str]

class SkillsListCompareRequest(BaseModel):
    list1: Dict[str, str] = Field(..., description="First skills list {skill: description}")
    list2: Dict[str, str] = Field(..., description="Second skills list {skill: description}")



resume_parser = IntelligentResumeParser()
job_parser = IntelligentFreelanceJobParser()
profile_builder = ProfileBuilder()
transformer = JobDescriptionTransformer()
graph = OllamaGraph(db_path="sqlite:///job_graph.db", output_path="job_tasks_graph.json")
intent_analyzer = JobIntentAnalyzer()
comparison_analyzer = JobComparisonAnalyzer()
skill_validator = JobSkillValidator()
foundation_divergence_analyzer = JobSkillComparator()
skills_list_comparator = SkillsListComparator()

@app.get("/")
def root():
    return {"status": "ok", "message": "Job Parser API is running"}


@app.post("/parse")
def parse_responsibility(req: ResumeRequest):
    try:
        return {"result": resume_parser.analyze_responsibilities(req.paragraph)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_jobs")
def analyze_jobs(req: JobRequest):
    try:
        results = job_parser.analyze(req.text.dict())
        return {"jobs": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/build_profiles")
def build_profiles(req: ProfileRequest):
    try:
        jobs = req.jobs if isinstance(req.jobs, list) else [req.jobs]
        results = profile_builder.analyze_jobs(jobs)
        return {"profiles": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph/ingest_subtasks")
def ingest_subtasks(payload: IngestSubtasksRequest):
    try:
        subtasks = [item.dict() for item in payload.subtasks]
        results = graph.process_main_actions(subtasks)
        return {
            "status": "ok",
            "ingested": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/tasks_by_group")
def tasks_by_group(name: str = Query(..., min_length=1, description="Exact group name")):
    try:
        tasks = graph.get_tasks_by_group(name)
        return {"group": name, "count": len(tasks), "tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/export_graph")
def export_graph():
    path = graph.save_to_json()
    return {"message": f"Graph exported to {path}"}



@app.post("/analyze_intent")
def analyze_intent(req: IntentAnalysisRequest):

    try:

        job_data = {
            "description": req.description,
            #"title": req.title or "Untitled Job"
        }

        # Analyze the job
        results = intent_analyzer.analyze(job_data)

        return {
            "status": "ok",
            "analysis": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transform_job")
def transform_job(req: IntentAnalysisRequest):
    try:
        job_data = {"description": req.description}
        results = transformer.transform(job_data)
        return {"status": "ok", "transformation": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare_jobs")
def compare_jobs(req: JobCompareRequest):
    try:
        results = comparison_analyzer.compare(
            {"description": req.job1},
            {"description": req.job2}
        )
        return {
            "status": "ok",
            "comparison": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate_job_skills")
def validate_job_skills(req: JobSkillValidationRequest):
    try:
        results = skill_validator.validate(
            job=req.job,
            skills=req.skills,
            strictness=req.strictness
        )
        return {
            "status": "ok",
            "validation": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare_job_skills")
def compare_job_skills(req: JobSkillComparisonRequest):
    try:
        results = foundation_divergence_analyzer.compare(
            {"description": req.job},
            req.skills
        )
        return {"status": "ok", "comparison": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare_skills_lists")
def compare_skills_lists(req: SkillsListCompareRequest):

    try:
        results = skills_list_comparator.compare(req.list1, req.list2)
        return {"status": "ok", "comparison": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
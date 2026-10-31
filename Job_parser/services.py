from dataclasses import dataclass
from typing import Optional

from Job_parser.config import Settings
from Job_parser.core.Job_Intent_clientExpertise import JobIntentAnalyzer
from Job_parser.core.freelance_parser import IntelligentFreelanceJobParser
from Job_parser.core.knowledge_graph import OllamaGraph
from Job_parser.core.parser import IntelligentResumeParser
from Job_parser.core.profile_builder import ProfileBuilder
from Job_parser.llm import LLMClient
from Job_parser.New_pipeline.Job_Skills_List_Comparison import JobSkillComparator
from Job_parser.New_pipeline.Job_cleaning import JobDescriptionTransformer
from Job_parser.New_pipeline.Job_to_job_skill_comparison import JobComparisonAnalyzer
from Job_parser.New_pipeline.Job_vs_skill_list import JobSkillValidator
from Job_parser.New_pipeline.Skills_comparison import SkillsListComparator


@dataclass
class Services:
    fast_llm: LLMClient
    reasoning_llm: LLMClient
    resume_parser: IntelligentResumeParser
    job_parser: IntelligentFreelanceJobParser
    profile_builder: ProfileBuilder
    graph: OllamaGraph
    intent_analyzer: JobIntentAnalyzer
    transformer: JobDescriptionTransformer
    job_comparator: JobComparisonAnalyzer
    skill_validator: JobSkillValidator
    job_skill_comparator: JobSkillComparator
    skills_list_comparator: SkillsListComparator


def build_services(
    settings: Settings,
    fast_llm: Optional[LLMClient] = None,
    reasoning_llm: Optional[LLMClient] = None,
) -> Services:
    """Wire every analyzer to its LLM. The `core/` analyzers use the fast model, `New_pipeline/` the reasoning model."""
    fast_llm = fast_llm or LLMClient.from_settings(settings, settings.fast_model)
    reasoning_llm = reasoning_llm or LLMClient.from_settings(settings, settings.reasoning_model)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    return Services(
        fast_llm=fast_llm,
        reasoning_llm=reasoning_llm,
        resume_parser=IntelligentResumeParser(fast_llm),
        job_parser=IntelligentFreelanceJobParser(fast_llm, db_url=settings.jobs_db_url),
        profile_builder=ProfileBuilder(fast_llm),
        graph=OllamaGraph(fast_llm, db_path=settings.graph_db_url, output_path=str(settings.graph_export_path)),
        intent_analyzer=JobIntentAnalyzer(fast_llm),
        transformer=JobDescriptionTransformer(reasoning_llm),
        job_comparator=JobComparisonAnalyzer(reasoning_llm),
        skill_validator=JobSkillValidator(reasoning_llm),
        job_skill_comparator=JobSkillComparator(reasoning_llm),
        skills_list_comparator=SkillsListComparator(reasoning_llm),
    )

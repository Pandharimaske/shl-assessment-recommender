from typing import List, Optional
from pydantic import BaseModel, Field

class AnalyzeOutput(BaseModel):
    # Safety and flow
    verdict: str = Field(..., description="ALLOWED (continue), BLOCKED (unrelated/safety), or EOC (finalize shortlist)")
    
    # Intent extraction
    job_role: Optional[str] = Field(None, description="The role being hired for")
    seniority: Optional[str] = Field(None, description="Seniority level")
    skills: List[str] = Field(default_factory=list, description="Specific skills mentioned")
    test_type_hints: List[str] = Field(default_factory=list, description="Preferred test types")
    purpose: Optional[str] = Field(None, description="hiring or development")
    ready_to_recommend: bool = Field(False, description="True if we have enough info to recommend")
    jd_provided: bool = Field(False, description="True if a full job description was provided")
    explicit_removes: List[str] = Field(default_factory=list, description="Assessments user explicitly rejected")
    explicit_adds: List[str] = Field(default_factory=list, description="Assessments user explicitly requested")
    languages: List[str] = Field(default_factory=list, description="Specific language requirements")
    industry: Optional[str] = Field(None, description="Industry context")
    hyde_description: Optional[str] = Field(None, description="Hypothetical assessment description")

class RerankOutput(BaseModel):
    reply: str = Field(..., description="Natural language explanation of the recommendations")
    recommendation_urls: List[str] = Field(..., description="List of EXACT URLs from the provided catalog items")
    end_of_conversation: bool = Field(False, description="True if the task is considered complete")

class SimpleOutput(BaseModel):
    reply: str = Field(..., description="The assistant's natural language response")

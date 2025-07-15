from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class SourceMetadata(BaseModel):
    """Simplified metadata with only core fields"""
    
    #source_type: Optional[str] = None
    doc_ref_id: Optional[str] = None
    score: Optional[float] = None
    
    DOC_DESCRIPTION: Optional[str] = None
    DOC_TITLE: Optional[str] = None
    DOC_DESCRIPTION_FORMATTED: Optional[str] = None
    TAGS: Optional[str] = None
    PRESENTATION_DATE: Optional[str] = None
    DOC_MODULE: Optional[str] = None
    PRESENTATION_LINK: Optional[str] = None
    PRESENTER_1_NAME: Optional[str] = None

class SourceNode(BaseModel):
    """Enhanced source node with metadata"""
    text: str = Field(description="The text content of the source")
    score: float = Field(description="Relevance score")
    metadata: SourceMetadata = Field(description="Source metadata")
    node_id: Optional[str] = None

class AIResponse(BaseModel):
    """Structured response from AI models"""
    answer: str = Field(description="The main answer to the user's question")
    suggestions: List[str] = Field(description="Additional suggestions or related information")
    was_context_valid: bool = Field(description="Whether the provided context was sufficient to answer the question")
    confidence_score: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)

class SearchRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    response_mode: Optional[str] = "compact"
    is_follow_up: bool = False
    previous_context: Optional[str] = None


class SearchResponse(BaseModel):
    answer: str
    context: str
    suggestions: List[str]
    was_context_valid: bool
    confidence_score: float
    success: bool
    ai_used: str
    processing_time: float
    source_nodes: List[SourceNode] = Field(description="Source nodes with metadata")
    synthesis_method: str = Field(description="Method used for response synthesis")
    total_sources: int = Field(description="Total number of sources used")

class RequestLog(BaseModel):
    """Request logging model"""
    request_id: str
    timestamp: datetime
    query: str
    cleaned_query: str
    top_k: int
    response_mode: str
    # context: str
    chunks: List[dict]
    gemini_input: dict
    gemini_output: dict
    final_response: dict
    processing_time: float
    success: bool
    error: Optional[str] = None


class ChatMessage(BaseModel):
    """Individual chat message"""
    role: str = Field(description="Either 'user' or 'model'")
    content: str = Field(description="Message content")
    
class ChatSearchRequest(BaseModel):
    """Enhanced search request for chat functionality"""
    question: str
    top_k: Optional[int] = 3
    response_mode: Optional[str] = "compact"
    message_history: Optional[List[ChatMessage]] = []
    previous_context: Optional[str] = ""
    

class ChatSearchResponse(BaseModel):
    """Enhanced response for chat functionality"""
    answer: str
    context: str
    suggestions: List[str]
    was_context_valid: bool
    was_context_valid_new_key: bool
    was_context_valid_old_key: bool
    confidence_score: float
    success: bool
    ai_used: str
    processing_time: float
    source_nodes: List[SourceNode] = Field(description="Source nodes with metadata")
    synthesis_method: str = Field(description="Method used for response synthesis")
    total_sources: int = Field(description="Total number of sources used")
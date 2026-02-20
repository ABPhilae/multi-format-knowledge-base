"""
Data models for the Multi-Format Knowledge Base.

These Pydantic models define the exact shape of data flowing in and out
of every endpoint. Same pattern as Phase 1 and Phase 2 — but now we have
models for document processing status, evaluation results, and chat
history in addition to the basic question/answer pattern.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


# ================================================================
# ENUMS
# ================================================================

class ProcessingStatus(str, Enum):
    """Status of a document ingestion job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileType(str, Enum):
    """Supported file types."""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    TXT = "txt"


# ================================================================
# REQUEST MODELS
# ================================================================

class QuestionRequest(BaseModel):
    """A question about the uploaded documents."""
    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Natural language question about the documents"
    )
    use_reranking: bool = Field(
        default=True,
        description="Whether to apply Cohere re-ranking (slower but better)"
    )
    top_k: int = Field(
        default=5,
        description="Number of source chunks to return"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "question": "What are the critical findings in the risk assessment?",
                "use_reranking": True,
                "top_k": 5
            }]
        }
    }


class ChatRequest(BaseModel):
    """A message in a conversation (supports follow-up questions)."""
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(
        default=None,
        description="ID for continuing a conversation. None starts a new one."
    )


# ================================================================
# RESPONSE MODELS
# ================================================================

class SourceChunk(BaseModel):
    """A single retrieved source chunk."""
    content: str
    source: str  # File name
    file_type: str
    page: Optional[int] = None
    sheet_name: Optional[str] = None
    slide_number: Optional[int] = None
    relevance_score: Optional[float] = None


class AnswerResponse(BaseModel):
    """Response to a question."""
    answer: str
    sources: list[SourceChunk]
    model_used: str
    processing_time_ms: float


class ChatResponse(BaseModel):
    """Response in a conversation."""
    answer: str
    sources: list[SourceChunk]
    conversation_id: str


class DocumentInfo(BaseModel):
    """Information about an uploaded document."""
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    status: ProcessingStatus
    uploaded_at: str


class IngestionResponse(BaseModel):
    """Response after uploading a document."""
    job_id: str
    filename: str
    status: ProcessingStatus
    message: str


class DashboardStats(BaseModel):
    """Statistics for the dashboard."""
    total_documents: int
    total_chunks: int
    documents_by_type: dict[str, int]
    collection_name: str


class EvaluationResult(BaseModel):
    """RAGAS evaluation results."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float
    questions_evaluated: int
    timestamp: str

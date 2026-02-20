"""
Multi-Format Knowledge Base — FastAPI Backend.

This is the main application that ties together all services:
- Document upload and processing (multi-format)
- Question answering (LangChain RAG + LlamaIndex)
- Conversation memory (multi-turn chat)
- RAGAS evaluation
- Dashboard statistics

All the intelligence is in the services/ folder. This file just
connects HTTP endpoints to service methods — exactly like the
"waiter" analogy from Phase 1.
"""
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import os
import uuid
import time
import logging

from src.config import settings
from src.models import (
    QuestionRequest,
    ChatRequest,
    AnswerResponse,
    ChatResponse,
    IngestionResponse,
    DocumentInfo,
    DashboardStats,
    ProcessingStatus,
)
from src.services.document_processor import DocumentProcessor
from src.services.langchain_rag import LangChainRAGService
from src.services.llamaindex_engine import LlamaIndexEngine
from src.services.retrieval_service import AdvancedRetrievalService
from src.services.evaluation_service import EvaluationService
from src.services.memory_service import MemoryService

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================
# Service instances (created at startup)
# ================================================================
document_processor: DocumentProcessor = None
rag_service: LangChainRAGService = None
llama_engine: LlamaIndexEngine = None
retrieval_service: AdvancedRetrievalService = None
evaluation_service: EvaluationService = None
memory_service: MemoryService = None

# Track document processing jobs
processing_jobs: dict = {}
# Track uploaded document metadata
document_registry: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, cleanup on shutdown."""
    global document_processor, rag_service, llama_engine
    global retrieval_service, evaluation_service, memory_service

    logger.info("Initializing services...")

    # Initialize services
    document_processor = DocumentProcessor()
    retriever = document_processor.get_retriever()
    rag_service = LangChainRAGService(retriever)
    llama_engine = LlamaIndexEngine()
    retrieval_service = AdvancedRetrievalService(retriever)
    evaluation_service = EvaluationService(
        rag_ask_fn=lambda q: rag_service.ask(q)["answer"],
        retriever=retriever,
    )
    memory_service = MemoryService()

    # Ensure upload directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)

    logger.info("All services initialized successfully")
    yield
    logger.info("Shutting down services...")


# Create the FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multi-format document intelligence with LangChain, LlamaIndex, and RAGAS evaluation",
    lifespan=lifespan,
)

# CORS — allow Streamlit frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# HEALTH CHECK
# ================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint — used by Docker and monitoring."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "services": {
            "qdrant": document_processor is not None,
            "rag": rag_service is not None,
            "evaluation": evaluation_service is not None,
        }
    }


# ================================================================
# DOCUMENT UPLOAD AND MANAGEMENT
# ================================================================

def _process_document_background(job_id: str, file_path: str, filename: str):
    """Background task for document ingestion."""
    try:
        processing_jobs[job_id] = ProcessingStatus.PROCESSING
        result = document_processor.ingest_file(file_path)

        document_registry[job_id] = {
            "document_id": job_id,
            "filename": filename,
            "file_type": os.path.splitext(filename)[1].lstrip("."),
            "chunk_count": result["chunk_count"],
            "status": ProcessingStatus.COMPLETED,
            "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        processing_jobs[job_id] = ProcessingStatus.COMPLETED
        logger.info(f"Document {filename} processed: {result['chunk_count']} chunks")

    except Exception as e:
        processing_jobs[job_id] = ProcessingStatus.FAILED
        logger.error(f"Document processing failed for {filename}: {e}")


@app.post("/documents/upload", response_model=IngestionResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a document for processing.

    The file is saved immediately, and processing happens in the background.
    Use the /documents/{job_id}/status endpoint to check progress.

    Supports: PDF, DOCX, XLSX, PPTX, TXT
    """
    # Validate file type
    ext = os.path.splitext(file.filename)[1].lower()
    supported = {".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md"}
    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(supported)}"
        )

    # Save the file
    job_id = str(uuid.uuid4())
    file_path = os.path.join(settings.upload_dir, f"{job_id}_{file.filename}")
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Queue background processing
    processing_jobs[job_id] = ProcessingStatus.PENDING
    background_tasks.add_task(
        _process_document_background, job_id, file_path, file.filename
    )

    return IngestionResponse(
        job_id=job_id,
        filename=file.filename,
        status=ProcessingStatus.PENDING,
        message="Document uploaded successfully. Processing in background.",
    )


@app.get("/documents/{job_id}/status")
async def check_processing_status(job_id: str):
    """Check the processing status of an uploaded document."""
    status = processing_jobs.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": status}


@app.get("/documents")
async def list_documents():
    """List all uploaded and processed documents."""
    return list(document_registry.values())


# ================================================================
# QUESTION ANSWERING
# ================================================================

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the uploaded documents.

    Uses the LangChain RAG pipeline with optional re-ranking.
    """
    if request.use_reranking:
        # Use advanced retrieval with re-ranking
        docs = retrieval_service.retrieve(request.question)
        # Build context manually and ask
        context = LangChainRAGService._format_docs(docs)
        answer = rag_service.chain.invoke(request.question)
    else:
        result = rag_service.ask(request.question)
        return AnswerResponse(**result)

    result = rag_service.ask(request.question)
    return AnswerResponse(**result)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the knowledge base (supports follow-up questions).

    Send a conversation_id to continue a conversation, or leave it
    null to start a new one.
    """
    # Get or create conversation
    conv_id = memory_service.get_or_create_conversation(request.conversation_id)

    # Get chat history
    history = memory_service.get_history(conv_id)

    # Build conversational chain
    retriever = document_processor.get_retriever()
    chain = memory_service.build_conversational_chain(
        retriever, LangChainRAGService._format_docs
    )

    # Run with history
    answer = chain.invoke({
        "question": request.message,
        "chat_history": history,
    })

    # Save exchange
    memory_service.add_exchange(conv_id, request.message, answer)

    # Get sources
    docs = retriever.invoke(request.message)
    sources = [
        {
            "content": doc.page_content[:300],
            "source": doc.metadata.get("filename", "Unknown"),
            "file_type": doc.metadata.get("file_type", "unknown"),
        }
        for doc in docs[:5]
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        conversation_id=conv_id,
    )


# ================================================================
# EVALUATION
# ================================================================

@app.post("/evaluate")
async def run_evaluation():
    """
    Run RAGAS evaluation on the test dataset.

    This takes several minutes because it asks 20 questions and
    evaluates each answer. Results include scores for faithfulness,
    answer relevancy, context precision, and context recall.
    """
    results = evaluation_service.run_evaluation()
    return results


# ================================================================
# DASHBOARD
# ================================================================

@app.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get statistics for the dashboard page."""
    db_stats = document_processor.get_stats()

    # Count documents by type
    docs_by_type: dict = {}
    for doc in document_registry.values():
        ft = doc["file_type"]
        docs_by_type[ft] = docs_by_type.get(ft, 0) + 1

    return DashboardStats(
        total_documents=len(document_registry),
        total_chunks=db_stats["total_chunks"],
        documents_by_type=docs_by_type,
        collection_name=db_stats["collection_name"],
    )

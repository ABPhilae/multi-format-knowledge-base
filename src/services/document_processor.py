"""
Document Processor — The Ingestion Pipeline.

This service handles the full journey from uploaded file to searchable chunks:
  1. Load: Route to the correct file-type loader (Step 3)
  2. Split: Chunk the text using LangChain's RecursiveCharacterTextSplitter
  3. Embed & Store: Send chunks to Qdrant with embeddings

Analogy: This is the KITCHEN PREP station. Raw ingredients (files) come in.
They get washed (loaded), chopped (chunked), and stored in labelled
containers (vector DB) ready for the chef (LLM) to use when an order comes in.

Why RecursiveCharacterTextSplitter?
- It tries paragraph breaks first, then sentences, then words
- Chunks almost always end at natural boundaries
- This is the production standard — the same splitter used by most
  LangChain RAG applications
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from src.loaders import load_document
from src.config import settings
import logging
import uuid
import os

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Orchestrates the full document ingestion pipeline.

    Usage:
        processor = DocumentProcessor()
        result = processor.ingest_file("/path/to/document.pdf")
    """

    def __init__(self):
        # Embeddings — same model as Phase 2, wrapped by LangChain
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )

        # Text splitter — the LangChain standard
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            # Tries paragraph breaks first, then sentences, then words
        )

        # Qdrant client — the vector database
        self.qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        # Ensure the collection exists
        self._ensure_collection()

        # LangChain Qdrant wrapper — gives us the .as_retriever() interface
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=settings.collection_name,
            embedding=self.embeddings,
        )

    def _ensure_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if settings.collection_name not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=settings.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # text-embedding-3-small dimension
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {settings.collection_name}")

    def ingest_file(self, file_path: str) -> dict:
        """
        Full ingestion pipeline: load → chunk → embed → store.

        Args:
            file_path: Path to the document to ingest

        Returns:
            Dict with filename, chunk_count, and status
        """
        filename = os.path.basename(file_path)
        logger.info(f"Starting ingestion: {filename}")

        # Step 1: Load the document using the file-type router
        documents = load_document(file_path)
        logger.info(f"  Loaded {len(documents)} document sections from {filename}")

        # Step 2: Chunk the documents
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"  Split into {len(chunks)} chunks")

        # Step 3: Add the filename to every chunk's metadata
        # (important for source citations later)
        for chunk in chunks:
            chunk.metadata["filename"] = filename

        # Step 4: Embed and store in Qdrant
        # LangChain's add_documents handles embedding automatically
        self.vector_store.add_documents(chunks)
        logger.info(f"  Stored {len(chunks)} chunks in Qdrant")

        return {
            "filename": filename,
            "chunk_count": len(chunks),
            "status": "completed",
        }

    def get_retriever(self, search_kwargs: dict = None):
        """
        Get a LangChain retriever from the vector store.

        This is the bridge between storage and retrieval —
        the retriever is what chains and query engines use to find
        relevant chunks.
        """
        default_kwargs = {"k": settings.retrieval_top_k}
        if search_kwargs:
            default_kwargs.update(search_kwargs)
        return self.vector_store.as_retriever(search_kwargs=default_kwargs)

    def get_stats(self) -> dict:
        """Get collection statistics for the dashboard."""
        try:
            collection_info = self.qdrant_client.get_collection(settings.collection_name)
            return {
                "total_chunks": collection_info.points_count,
                "collection_name": settings.collection_name,
            }
        except Exception:
            return {"total_chunks": 0, "collection_name": settings.collection_name}

"""
LlamaIndex Query Engine Service.

LlamaIndex provides an alternative query pathway that excels at:
- Multi-document queries (comparing across files)
- Structured retrieval with node relationships
- Built-in response synthesis (not just "stuff the prompt")

In this project, LlamaIndex handles complex queries while LangChain
handles standard question-answering. They share the same vector store
(Qdrant), so there is no data duplication.

Connection to Phase 2:
In Phase 2, you had ONE retrieval path. Now you have TWO:
- LangChain LCEL chain (Step 4) — fast, streaming, conversational
- LlamaIndex query engine (this file) — better at synthesis and
  multi-document reasoning

The FastAPI endpoints will let the user choose, or auto-select based
on question complexity.
"""
from llama_index.core import (
    VectorStoreIndex,
    Settings as LlamaSettings,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode
from src.config import settings
import logging

logger = logging.getLogger(__name__)


class LlamaIndexEngine:
    """
    LlamaIndex-based query engine for advanced document queries.

    This wraps LlamaIndex's VectorStoreIndex and query engine
    for use alongside the LangChain pipeline.
    """

    def __init__(self):
        # Configure LlamaIndex global settings
        LlamaSettings.llm = LlamaOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        LlamaSettings.embed_model = OpenAIEmbedding(
            model_name=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

        self.node_parser = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        self.index = None
        self.query_engine = None

    def build_index(self, documents: list[dict]):
        """
        Build a LlamaIndex VectorStoreIndex from processed documents.

        Args:
            documents: List of dicts with 'content' and 'metadata' keys
        """
        # Convert to LlamaIndex TextNode objects
        nodes = []
        for doc in documents:
            node = TextNode(
                text=doc["content"],
                metadata=doc.get("metadata", {}),
            )
            nodes.append(node)

        # Build the index
        self.index = VectorStoreIndex(nodes)
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",  # Synthesises a concise answer
        )
        logger.info(f"Built LlamaIndex index with {len(nodes)} nodes")

    def query(self, question: str) -> dict:
        """
        Query the LlamaIndex index.

        Args:
            question: Natural language question

        Returns:
            Dict with answer and source_nodes
        """
        if self.query_engine is None:
            return {
                "answer": "No documents have been indexed yet. Please upload documents first.",
                "sources": [],
            }

        response = self.query_engine.query(question)

        sources = []
        for node in response.source_nodes:
            sources.append({
                "content": node.text[:300] + "..." if len(node.text) > 300 else node.text,
                "source": node.metadata.get("filename", "Unknown"),
                "file_type": node.metadata.get("file_type", "unknown"),
                "relevance_score": round(node.score, 4) if node.score else None,
            })

        return {
            "answer": str(response),
            "sources": sources,
        }

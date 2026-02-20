"""
Advanced Retrieval Service — Re-Ranking and Multi-Query.

This service wraps the base retriever with advanced techniques:
1. Multi-Query: Generates multiple reformulations of the question
2. Re-Ranking: Uses Cohere's model to re-score results by true relevance

Why these two techniques specifically?
- Multi-Query closes the "vocabulary gap" (user says "deadline", document
  says "remediation date" — one of the reformulations will match)
- Re-Ranking fixes the "similarity ≠ relevance" problem (the most similar
  chunk is not always the most USEFUL chunk for answering the question)

Together they typically improve RAG quality by 15-25%.

Analogy: The Search Engine Upgrade
Phase 2 was like early Google — it found pages with similar words.
Multi-query is like Google suggesting "Did you mean...?" and searching
all variations.
Re-ranking is like Google's PageRank — sorting results by actual
usefulness, not just keyword overlap.
"""
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from src.config import settings
import logging

logger = logging.getLogger(__name__)


class AdvancedRetrievalService:
    """
    Wraps a base retriever with multi-query and re-ranking.

    Usage:
        service = AdvancedRetrievalService(base_retriever)
        docs = service.retrieve("What are the HK findings?", use_reranking=True)
    """

    def __init__(self, base_retriever):
        """
        Args:
            base_retriever: A LangChain retriever (from DocumentProcessor)
        """
        self.base_retriever = base_retriever

        # Multi-query retriever: generates 3 alternative phrasings
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=ChatOpenAI(
                model=settings.openai_model,
                openai_api_key=settings.openai_api_key,
                temperature=0.3,  # Some creativity for reformulations
            ),
        )

        # Cohere re-ranker
        self.reranker = None
        if settings.cohere_api_key:
            try:
                self.reranker = CohereRerank(
                    model="rerank-english-v3.0",
                    top_n=settings.rerank_top_n,
                    cohere_api_key=settings.cohere_api_key,
                )
                logger.info("Cohere re-ranker initialized successfully")
            except Exception as e:
                logger.warning(f"Cohere re-ranker not available: {e}")

    def retrieve(
        self,
        question: str,
        use_reranking: bool = True,
        use_multi_query: bool = True,
    ) -> list[Document]:
        """
        Retrieve relevant documents with optional advanced techniques.

        The pipeline is:
        1. (Optional) Multi-query generates 3 reformulations
        2. Base retriever finds top-k candidates for each query
        3. Results are deduplicated
        4. (Optional) Re-ranker re-scores and returns top-n

        Args:
            question: The user's question
            use_reranking: Whether to apply Cohere re-ranking
            use_multi_query: Whether to use multi-query reformulation

        Returns:
            List of relevant Document objects, ranked by relevance
        """
        # Step 1: Retrieve candidates
        if use_multi_query:
            try:
                docs = self.multi_query_retriever.invoke(question)
                logger.info(f"Multi-query retrieved {len(docs)} unique documents")
            except Exception as e:
                logger.warning(f"Multi-query failed, falling back to base: {e}")
                docs = self.base_retriever.invoke(question)
        else:
            docs = self.base_retriever.invoke(question)

        # Step 2: Re-rank if enabled and available
        if use_reranking and self.reranker and len(docs) > 0:
            try:
                rerank_retriever = ContextualCompressionRetriever(
                    base_compressor=self.reranker,
                    base_retriever=self.base_retriever,
                )
                docs = rerank_retriever.invoke(question)
                logger.info(f"Re-ranked to {len(docs)} documents")
            except Exception as e:
                logger.warning(f"Re-ranking failed, using original order: {e}")

        return docs

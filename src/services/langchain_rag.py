"""
LangChain RAG Service — The Core Question-Answering Engine.

This is the heart of the system. It takes a question, retrieves relevant
chunks, and generates an answer using an LLM.

Built with LCEL (LangChain Expression Language) — the modern way to build
chains. LCEL uses the pipe operator (|) to connect components:

    prompt | llm | output_parser

This reads left to right: the prompt receives variables, formats itself,
passes to the LLM, whose output passes to the output parser.

Connection to Phase 2:
Your Phase 2 rag_service.py did the same thing manually:
  1. Embed the question
  2. Search ChromaDB
  3. Build a prompt with context
  4. Call OpenAI
  5. Return the answer

LangChain packages steps 1-5 into a single chain that also supports
streaming, memory, and retry logic.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from src.config import settings
import logging
import time

logger = logging.getLogger(__name__)

# The RAG prompt template — same idea as Phase 2, but formatted for LangChain
RAG_PROMPT = ChatPromptTemplate.from_template("""You are a helpful knowledge base assistant for an enterprise document library.
Answer the question based ONLY on the following context from the uploaded documents.
If the context does not contain enough information to answer, say so clearly.
Always cite which document and section your answer comes from.

Context from documents:
{context}

Question: {question}

Answer:""")


class LangChainRAGService:
    """
    RAG service built with LangChain LCEL.

    This replaces the manual pipeline from Phase 2 with professional
    components while keeping the same logic.
    """

    def __init__(self, retriever):
        """
        Args:
            retriever: A LangChain retriever (from DocumentProcessor)
        """
        self.retriever = retriever

        # The LLM — same as Phase 2, wrapped by LangChain
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,  # Deterministic answers for knowledge base
            openai_api_key=settings.openai_api_key,
            max_tokens=settings.max_tokens,
        )

        # Build the LCEL chain
        # This is the modern LangChain way — read left to right with |
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        """
        Format retrieved documents into a single context string.

        Each chunk is labelled with its source file and page/sheet
        so the LLM can cite them in its answer.
        """
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", doc.metadata.get("source", "Unknown"))
            file_type = doc.metadata.get("file_type", "unknown")

            # Build a source label based on file type
            label_parts = [f"[Source {i}: {source}"]
            if file_type == "pdf" and "page" in doc.metadata:
                label_parts.append(f"Page {doc.metadata['page'] + 1}")
            elif file_type == "xlsx" and "sheet_name" in doc.metadata:
                label_parts.append(f"Sheet: {doc.metadata['sheet_name']}")
            elif file_type == "pptx" and "slide_number" in doc.metadata:
                label_parts.append(f"Slide {doc.metadata['slide_number']}")
            label = ", ".join(label_parts) + "]"

            formatted.append(f"{label}\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted)

    def ask(self, question: str) -> dict:
        """
        Ask a question and get an answer with sources.

        Args:
            question: Natural language question

        Returns:
            Dict with answer, sources, model_used, and processing_time_ms
        """
        start_time = time.time()

        # Retrieve the source documents (for returning to the user)
        retrieved_docs = self.retriever.invoke(question)

        # Run the chain
        answer = self.chain.invoke(question)

        processing_time = (time.time() - start_time) * 1000

        # Format sources for the response
        sources = []
        for doc in retrieved_docs[:5]:  # Return top 5 sources
            sources.append({
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "source": doc.metadata.get("filename", "Unknown"),
                "file_type": doc.metadata.get("file_type", "unknown"),
                "page": doc.metadata.get("page"),
                "sheet_name": doc.metadata.get("sheet_name"),
                "slide_number": doc.metadata.get("slide_number"),
            })

        return {
            "answer": answer,
            "sources": sources,
            "model_used": settings.openai_model,
            "processing_time_ms": round(processing_time, 2),
        }

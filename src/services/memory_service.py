"""
Conversation Memory Service.

Phase 2 RAG was stateless — each question started fresh.
This service adds conversation memory so the AI remembers context:

User: "What are the critical findings in Hong Kong?"
AI: "The critical findings include..."
User: "Who is responsible for fixing them?"  ← THIS NOW WORKS
AI: "According to the tracker, Li Wei is responsible..."

The memory stores the last 5 exchanges per conversation and passes them
to the LLM alongside the retrieved context.

We use a simple in-memory dict for this project. In production, you would
use Redis (which is already in our Docker Compose for caching).
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import settings
import uuid
import logging

logger = logging.getLogger(__name__)

# Maximum number of exchanges to remember per conversation
MAX_HISTORY = 5


# Prompt that includes conversation history
CONVERSATIONAL_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful knowledge base assistant.
Answer the question based on the following context from uploaded documents.
If the context does not contain enough information, say so clearly.
Always cite which document your answer comes from.

Context from documents:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


class MemoryService:
    """
    Manages conversation history for multi-turn Q&A.

    Each conversation gets a unique ID. The history is stored in memory
    and passed to the LLM with each new question.
    """

    def __init__(self):
        # In-memory conversation store: {conversation_id: [messages]}
        self.conversations: dict[str, list] = {}

    def get_or_create_conversation(self, conversation_id: str = None) -> str:
        """Get existing conversation or create a new one."""
        if conversation_id and conversation_id in self.conversations:
            return conversation_id

        new_id = str(uuid.uuid4())
        self.conversations[new_id] = []
        logger.info(f"Created new conversation: {new_id}")
        return new_id

    def get_history(self, conversation_id: str) -> list:
        """Get the chat history as LangChain message objects."""
        history = self.conversations.get(conversation_id, [])
        messages = []
        for entry in history[-MAX_HISTORY:]:
            messages.append(HumanMessage(content=entry["question"]))
            messages.append(AIMessage(content=entry["answer"]))
        return messages

    def add_exchange(self, conversation_id: str, question: str, answer: str):
        """Record a question-answer exchange."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        self.conversations[conversation_id].append({
            "question": question,
            "answer": answer,
        })

        # Trim to MAX_HISTORY
        if len(self.conversations[conversation_id]) > MAX_HISTORY:
            self.conversations[conversation_id] = self.conversations[conversation_id][-MAX_HISTORY:]

    def build_conversational_chain(self, retriever, format_docs_fn):
        """
        Build an LCEL chain that includes conversation history.

        Args:
            retriever: LangChain retriever
            format_docs_fn: Function to format retrieved docs into text

        Returns:
            An LCEL chain that accepts question and chat_history
        """
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )

        chain = (
            {
                "context": lambda x: format_docs_fn(retriever.invoke(x["question"])),
                "chat_history": lambda x: x["chat_history"],
                "question": lambda x: x["question"],
            }
            | CONVERSATIONAL_RAG_PROMPT
            | llm
            | StrOutputParser()
        )

        return chain

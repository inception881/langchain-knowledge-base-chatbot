from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver # Keep async
import aiosqlite # Ensure import
import asyncio

from typing import List, Any, Optional
from pathlib import Path

from src.config import Config
from src.prompts.templates import PromptTemplate
from src.chat_model import get_chat_model
from src.loaders.document_loader import get_document_loader
from src.vectorstores.faiss_store import get_faiss_vector_store
from src.vectorstores.hybrid_retriever import create_hybrid_retriever
from src.utils.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

# Vector store path
FAISS_INDEX_PATH = Config.FAISS_INDEX_PATH
FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)

class RetrievalTool(BaseTool):
    # ... (Keep RetrievalTool code unchanged) ...
    name: str = "retrieval_tool"
    description: str = "Retrieve relevant documents from knowledge base"
    retriever: Any = None
    last_docs: List[Document] = []
    
    def __init__(self, retriever):
        super().__init__(retriever=retriever)
    
    def _run(self, query: str) -> str:
        # Synchronous call entry point (if called synchronously for some reason)
        # Note: In fully async Agents, LangChain typically looks for _arun first or runs _run in a thread pool
        return self._run_sync(query)

    async def _arun(self, query: str) -> str:
        """Async implementation of retrieval"""
        # Assuming retriever supports async invoke, if not, can use run_in_executor
        docs = await self.retriever.ainvoke(query) 
        self.last_docs = docs
        if not docs:
            return "No relevant documents found."
        formatted_docs = "\n\n".join([f"<doc>\n{doc.page_content}\n</doc>" for doc in docs])
        return formatted_docs

    def _run_sync(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        self.last_docs = docs
        if not docs:
            return "No relevant documents found."
        formatted_docs = "\n\n".join([f"<doc>\n{doc.page_content}\n</doc>" for doc in docs])
        return formatted_docs
    
    def get_last_docs(self) -> List[Document]:
        return self.last_docs

class FAISSConversationalRAGChain:
    """FAISS-based Conversational RAG Chain using Agent implementation"""
    
    def __init__(self, session_id: str = "default"):
        """
        Initialize strictly synchronous components.
        Async components are initialized in ainitialize().
        """
        self.session_id = session_id
        self.llm = get_chat_model()
        self.document_loader = get_document_loader()
        self.faiss_store = get_faiss_vector_store()
        self.retriever = create_hybrid_retriever(self.faiss_store.vector_store)
        self.prompt_template = PromptTemplate.template
        self.retrieval_tool = RetrievalTool(self.retriever)
        
        # Async resource placeholders
        self.agent = None
        self.db_conn = None
        
        # Key: Record the Event Loop that created the resources
        self._bound_loop = None 

    async def ainitialize(self):
        """
        Async initialization: Need to check if Loop has changed on each Streamlit Rerun
        """
        current_loop = asyncio.get_running_loop()

        # Check if reinitialization is needed
        # If already initialized and current Loop is the same as bound Loop, skip
        if self.agent is not None and self._bound_loop is current_loop:
            return

        logger.info(f"Async Initializing (Loop Changed/First Run) for session: {self.session_id}")

        # 1. Resource cleanup: If there was a previous connection from old Loop, try to clean up (although old Loop may be dead)
        if self.db_conn:
            try:
                # Only attempt to close if connection is not already closed
                await self.db_conn.close()
            except Exception:
                pass # Ignore errors when closing old connections

        # 1. MCP Client & Tools (Async)
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        # Your API Key
        TAVILY_KEY = Config.TAVILY_KEY

        client = MultiServerMCPClient(
            {
                "tavily": {
                    "transport": "streamable_http",  # Tavily is an HTTP MCP server
                    "url": "https://mcp.tavily.com/mcp/",
                    "headers": {
                        # Key modification points:
                        # 1. Most Tavily services expect header key to be "api-key" (without X-)
                        "api-key": TAVILY_KEY,
                        # 2. Or Content-Type
                        "Content-Type": "application/json",
                        # 3. To prevent certain gateway interceptions, can also add Bearer Token form (optional, double insurance)
                        "Authorization": f"Bearer {TAVILY_KEY}"
                    }
                }
            }
        )
        

        
        try:
            tavily_tools = await client.get_tools()
            logger.info(f"MCP Tools loaded: {len(tavily_tools)}")
        except Exception as e:
            logger.error(f"MCP Connection failed: {e}")
            tavily_tools = []

        # 3. Database connection (must be rebuilt in current Loop)
        db_path = Config.SHORT_TERM_MEMORY / f"{self.session_id}.db"
        self.db_conn = await aiosqlite.connect(str(db_path), check_same_thread=False)
        await self.db_conn.execute("PRAGMA journal_mode=WAL;")
        await self.db_conn.commit()

        # 4. Middleware
        from src.memory.long_term_memory import (
            retrieve_similar_history_middleware,
            save_assistant_response_middleware,
            save_user_messages_middleware,
            sanitize_dangling_tool_middleware
        )
        from src.memory.query_rewriter import query_rewriter_middleware

        # 5. Rebuild Agent (Agent's checkpointer must be bound to new connection)
        self.agent = create_agent(
            model=self.llm,
            tools=[self.retrieval_tool, *tavily_tools],
            middleware=[
                sanitize_dangling_tool_middleware,
                query_rewriter_middleware,

                SummarizationMiddleware(
                    model=get_chat_model(model="claude-sonnet-4-5"),
                    max_tokens_before_summary=4000,
                    messages_to_keep=20,
                ),
                retrieve_similar_history_middleware,
                save_user_messages_middleware,
                save_assistant_response_middleware
            ],
            # The conn here is created under the new Loop
            checkpointer=AsyncSqliteSaver(conn=self.db_conn),
        )
        
        # 6. Update bound Loop marker
        self._bound_loop = current_loop
        logger.info("Initialization complete on current event loop.")
 
    from streamlit.runtime.uploaded_file_manager import UploadedFile
    def add_documents(self, file_path: UploadedFile):
        """
        Add documents to vector store
        
        Args:
            file_path: File path
        """
        # file_path = self.document_loader.save_uploaded_file(file_path)
        # Convert string path to Path object
        # path = Path(file_path)
        
        # Process file
        chunks = self.document_loader._process_file(file_path=file_path, skip_processed=True)
        
        if not chunks:
            logger.warning(f"⚠️ File {file_path.name} did not generate any document chunks after processing")
            return {"message": f"Skipping already processed file:  {file_path.name} "}
        
        # Use FAISS vector store service to add documents
        self.faiss_store.add_documents(chunks)
        
        # # Update retriever
        # self.retriever = self.faiss_store.get_retriever()
    
    def  delete_documents(self, doc_id: str):
        """
        Delete document from vector store
        
        Args:
            doc_id: Document ID

        """
        self.document_loader.delete_processed_document(doc_id)
        self.faiss_store.delete_by_source(doc_id)
    
    def clear_documents(self):
        """Clear documents and conversation history"""
        # Clear documents from knowledge base
        self.document_loader.clear_all_processed_documents()
        self.faiss_store.clear()

# Session management
    async def aclose(self):
            """Cleanup async resources"""
            if self.db_conn:
                await self.db_conn.close()

# Session management
_sessions = {}

def get_conversational_chain(session_id: str) -> FAISSConversationalRAGChain:
    """Simple factory, does NOT initialize async parts yet."""
    if session_id not in _sessions:
        _sessions[session_id] = FAISSConversationalRAGChain(session_id)
    return _sessions[session_id]

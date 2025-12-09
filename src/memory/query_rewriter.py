import logging
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.middleware import before_model
from src.chat_model import get_chat_model

logger = logging.getLogger(__name__)

# System prompt for query rewriting
REWRITE_SYSTEM_PROMPT = """
You are a professional query optimization expert. Your task is to rewrite the user's input into a "standalone query" that is optimized for search engines and AI understanding.

Rules:
1. Contextualization: If the user's query contains pronouns like "it", "that", or "he", replace them with specific nouns based on the chat history.
2. Semantic Completion: If the user's query is incomplete (e.g., just "what about the price?"), complete the subject and predicate.
3. Noise Removal: Remove conversational fillers (e.g., "hey", "can you check", "please"), keeping only the core intent.
4. Maintain Intent: Do NOT answer the question. Do NOT change the user's core intent.
5. Temporal Handling: If the user asks about "today" or "now", keep these words as they are (time handling is done by other components).

Example:
Chat History: [User: "Tell me about the iPhone 15."]
User Input: "How is its battery?"
Output: "How is the battery life of the iPhone 15?"

Do not output any explanations. Only output the rewritten query string.
"""
@before_model
async def query_rewriter_middleware(state: dict, runtime):
    """
    Query Rewriter Middleware
    Intercepts and optimizes the user's HumanMessage before the Agent processes it.
    """
    messages = state.get("messages", [])
    
    # 1. Safety check: return if no messages
    if not messages:
        return state
    
    # 2. Get the last message
    last_message = messages[-1]
    
    # 3. Process only if the last message is from a human
    # Note: If we inserted a SystemMessage (e.g., for time) in web_chatbot.py,
    # ensure the HumanMessage is the actual last one in the list.
    if not isinstance(last_message, HumanMessage):
        return state

    original_query = last_message.content
    
    # Skip rewriting if the query is too short or clearly doesn't need context (e.g., "Hello")
    if len(original_query) < 2:
        return state

    logger.info(f"🔄 Rewriting Query: Raw input -> '{original_query}'")

    try:
        # 4. Prepare the rewriting model 
        # (Using temperature=0 ensures deterministic output)
        rewriter_llm = get_chat_model(temperature=0, model="claude-haiku-4-5")
        
        # 5. Build the Prompt Chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", REWRITE_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        chain = prompt | rewriter_llm | StrOutputParser()
        
        # Extract history (excluding the current last message)
        history = messages[:-1]
        
        # 6. Execute rewriting (Async)
        rewritten_query = await chain.ainvoke({
            "history": history,
            "question": original_query
        })
        
        # Strip whitespace
        rewritten_query = rewritten_query.strip()
        
        # 7. Critical Step: Update the message content in the state
        # The Agent will now see the optimized query instead of the raw one.
        if rewritten_query and rewritten_query != original_query:
            logger.info(f"✅ Query Rewritten: '{original_query}' -> '{rewritten_query}'")
            
            # Modify the content directly
            messages[-1].content = rewritten_query
            
            # Optional: Store original query in metadata for debugging
            if "original_query" not in messages[-1].additional_kwargs:
                messages[-1].additional_kwargs["original_query"] = original_query

    except Exception as e:
        logger.error(f"⚠️ Query rewrite failed: {e}")
        # If rewriting fails, proceed with the original query to avoid breaking the flow.
    
    # Middleware must return the state
    return state

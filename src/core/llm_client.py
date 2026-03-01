"""
Unified LLM client for the Product Success Tracking Agent.

Abstracts away the LLM provider (Gemini, OpenAI, Ollama) so the rest of
the codebase doesn't need to know which provider is being used.

Why this abstraction?
- Easy to swap providers by changing config
- Consistent interface for all agents
- Centralized error handling and retry logic
- **All LLM calls are traced to Langfuse** for observability
"""

import google.generativeai as genai
import time
from typing import Optional
from src.core.config import settings
from src.core.observability import get_langfuse, get_logger

logger = get_logger(__name__)


def get_gemini_client():
    """
    Initialize and return a Gemini client.
    
    Why configure here?
    - API key is loaded from settings (environment variable)
    - Single point of configuration for all Gemini calls
    """
    genai.configure(api_key=settings.gemini_api_key)
    return genai


def get_llm_model(model_name: Optional[str] = None):
    """
    Get a Gemini generative model instance.
    
    Args:
        model_name: Override the default model from settings
        
    Returns:
        A Gemini GenerativeModel instance ready for chat/generation
        
    Why return a model instance?
    - Each model instance can have its own system instructions
    - Allows different agents to use different models
    """
    # Initialize the client (configures API key)
    get_gemini_client()
    
    # Use provided model or fall back to settings
    model = model_name or settings.gemini_model
    
    return genai.GenerativeModel(model)


def get_embedding_model():
    """
    Get the embedding model name for vector operations.
    
    Why separate from LLM?
    - Embeddings use a different model optimized for semantic similarity
    - text-embedding-004 produces 768-dimensional vectors
    - Much cheaper than using the main LLM for embeddings
    """
    # Initialize the client (configures API key)
    get_gemini_client()
    
    return settings.gemini_embedding_model


async def generate_text(
    prompt: str,
    system_instruction: Optional[str] = None,
    model_name: Optional[str] = None
) -> str:
    """
    Generate text using Gemini.
    
    Args:
        prompt: The user prompt to send to the model
        system_instruction: Optional system prompt to guide behavior
        model_name: Override the default model
        
    Returns:
        Generated text response
        
    Why async?
    - LLM calls are I/O bound (waiting for API response)
    - Async allows other work to happen while waiting
    """
    # Get model with optional system instruction
    get_gemini_client()
    model = model_name or settings.gemini_model
    
    if system_instruction:
        llm = genai.GenerativeModel(
            model,
            system_instruction=system_instruction
        )
    else:
        llm = genai.GenerativeModel(model)
    
    # Generate response
    response = await llm.generate_content_async(prompt)
    
    return response.text


def generate_text_sync(
    prompt: str,
    system_instruction: Optional[str] = None,
    model_name: Optional[str] = None,
    trace_name: Optional[str] = None,
    trace_metadata: Optional[dict] = None
) -> str:
    """
    Synchronous version of generate_text with Langfuse tracing.
    
    Why both sync and async?
    - Some contexts (like CLI scripts) don't have an event loop
    - FastAPI endpoints can use async version
    
    Args:
        prompt: The user prompt to send to the model
        system_instruction: Optional system prompt to guide behavior
        model_name: Override the default model
        trace_name: Name for this trace in Langfuse (e.g., "context_discovery")
        trace_metadata: Additional context to log (e.g., {"feature": "AI Search"})
        
    Returns:
        Generated text response
        
    Observability:
        Every call is logged to Langfuse with:
        - Input prompt and system instruction
        - Output text
        - Model used
        - Latency in milliseconds
        - Any metadata you provide
    """
    get_gemini_client()
    model = model_name or settings.gemini_model
    
    # Build the model with optional system instruction
    if system_instruction:
        llm = genai.GenerativeModel(
            model,
            system_instruction=system_instruction
        )
    else:
        llm = genai.GenerativeModel(model)
    
    # Track timing for observability
    start_time = time.time()
    
    # Make the actual LLM call
    response = llm.generate_content(prompt)
    output_text = response.text
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log to Langfuse for observability
    # This is where you'll see the call in your Langfuse dashboard
    _trace_to_langfuse(
        name=trace_name or "llm_generation",
        model=model,
        input_text=prompt,
        system_instruction=system_instruction,
        output_text=output_text,
        latency_ms=latency_ms,
        metadata=trace_metadata
    )
    
    logger.info(
        "llm_call_complete",
        model=model,
        latency_ms=round(latency_ms, 2),
        trace_name=trace_name
    )
    
    return output_text


def _trace_to_langfuse(
    name: str,
    model: str,
    input_text: str,
    output_text: str,
    latency_ms: float,
    system_instruction: Optional[str] = None,
    metadata: Optional[dict] = None
):
    """
    Send trace data to Langfuse.
    
    Why a separate function?
    - Keeps the main generate function clean
    - Easy to modify tracing without touching generation logic
    - Gracefully handles missing Langfuse config
    
    What gets logged:
    - A "trace" groups related operations (like one pipeline run)
    - A "generation" is a single LLM call within that trace
    - You can see all this in the Langfuse dashboard
    """
    client = get_langfuse()
    if client is None:
        # Langfuse not configured, skip tracing
        return
    
    try:
        # Create a trace (top-level grouping)
        trace = client.trace(
            name=name,
            metadata=metadata or {}
        )
        
        # Add the generation (the actual LLM call)
        # This is what you'll see in the Langfuse "Generations" tab
        trace.generation(
            name=f"{name}_generation",
            model=model,
            input={
                "prompt": input_text,
                "system_instruction": system_instruction
            } if system_instruction else input_text,
            output=output_text,
            metadata={
                "latency_ms": round(latency_ms, 2),
                **(metadata or {})
            }
        )
        
        # Flush to ensure data is sent
        # Important: Langfuse batches requests, flush ensures immediate send
        client.flush()
        
    except Exception as e:
        # Don't fail the main operation if tracing fails
        logger.warning("langfuse_trace_error", error=str(e))


def embed_text(text: str) -> list[float]:
    """
    Generate embeddings for a piece of text.
    
    Args:
        text: The text to embed
        
    Returns:
        A list of floats representing the embedding vector (768 dimensions)
        
    Why embeddings?
    - Convert text to numerical vectors for similarity search
    - Similar texts have similar vectors (close in vector space)
    - This is the foundation of semantic search in RAG
    """
    get_gemini_client()
    
    result = genai.embed_content(
        model=f"models/{settings.gemini_embedding_model}",
        content=text,
        task_type="retrieval_document"
    )
    
    return result['embedding']


def embed_query(query: str) -> list[float]:
    """
    Generate embeddings for a search query.
    
    Why separate from embed_text?
    - Queries are typically shorter than documents
    - Using task_type="retrieval_query" optimizes for query embedding
    - This asymmetric approach improves retrieval quality
    """
    get_gemini_client()
    
    result = genai.embed_content(
        model=f"models/{settings.gemini_embedding_model}",
        content=query,
        task_type="retrieval_query"
    )
    
    return result['embedding']

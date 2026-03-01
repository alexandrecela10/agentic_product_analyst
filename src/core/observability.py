"""
Observability setup for the Product Success Tracking Agent.

Integrates Langfuse for LLM tracing and OpenTelemetry for general observability.

Why observability?
- Track LLM calls, latency, and costs
- Debug RAG retrieval quality
- Monitor agent behavior in production
- Essential for evaluating and improving the system
"""

import structlog
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from typing import Optional, Any
from src.core.config import settings


# --- Structured Logging ---
# Why structlog?
# - JSON-formatted logs for easy parsing
# - Contextual logging (add fields to all logs in a request)
# - Better than print() for production

def setup_logging():
    """
    Configure structured logging with structlog.
    
    Why configure once at startup?
    - Consistent log format across all modules
    - Processors run in order for each log event
    """
    structlog.configure(
        processors=[
            # Add timestamp to every log
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            # Format as JSON for production, pretty for dev
            structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    """
    Get a logger instance for a module.
    
    Args:
        name: Usually __name__ of the calling module
        
    Returns:
        A structlog logger bound to that name
    """
    return structlog.get_logger(name)


# --- Langfuse Integration ---
# Why Langfuse?
# - Purpose-built for LLM observability
# - Tracks prompts, completions, latency, costs
# - Supports evaluation and feedback
# - You already have an account!

_langfuse_client: Optional[Langfuse] = None


def get_langfuse() -> Optional[Langfuse]:
    """
    Get or create the Langfuse client singleton.
    
    Why singleton?
    - Only need one connection to Langfuse
    - Reuse across all LLM calls
    
    Returns:
        Langfuse client or None if not configured
    """
    global _langfuse_client
    
    # Skip if no keys configured
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    
    if _langfuse_client is None:
        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url  # Fixed: matches .env variable name
        )
    
    return _langfuse_client


def trace_llm_call(
    name: str,
    input_text: str,
    output_text: str,
    model: str,
    metadata: Optional[dict] = None
):
    """
    Manually trace an LLM call to Langfuse.
    
    Args:
        name: Name of the operation (e.g., "context_discovery")
        input_text: The prompt sent to the LLM
        output_text: The response from the LLM
        model: Model name used
        metadata: Additional context to log
        
    Why manual tracing?
    - Works with any LLM provider (Gemini, OpenAI, etc.)
    - Full control over what gets logged
    """
    client = get_langfuse()
    if client is None:
        return
    
    trace = client.trace(name=name, metadata=metadata or {})
    trace.generation(
        name=f"{name}_generation",
        model=model,
        input=input_text,
        output=output_text
    )


# --- OpenTelemetry Setup ---
# Why OpenTelemetry?
# - Industry standard for distributed tracing
# - Works with many backends (Jaeger, Zipkin, etc.)
# - Traces non-LLM operations (DB queries, HTTP calls)

def setup_opentelemetry():
    """
    Configure OpenTelemetry tracing.
    
    Why configure at startup?
    - Provider must be set before any traces are created
    - Console exporter for local dev, can swap for production
    """
    provider = TracerProvider()
    
    # For development: print spans to console
    if settings.debug:
        processor = SimpleSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
    
    trace.set_tracer_provider(provider)


def get_tracer(name: str):
    """
    Get an OpenTelemetry tracer for a module.
    
    Args:
        name: Usually __name__ of the calling module
        
    Returns:
        An OpenTelemetry tracer
    """
    return trace.get_tracer(name)


# --- Initialization ---
def init_observability():
    """
    Initialize all observability components.
    
    Call this once at application startup.
    """
    setup_logging()
    setup_opentelemetry()
    
    logger = get_logger(__name__)
    logger.info("observability_initialized", 
                langfuse_enabled=get_langfuse() is not None,
                debug_mode=settings.debug)

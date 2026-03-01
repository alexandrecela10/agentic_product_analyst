"""
Configuration management for the Product Success Tracking Agent.

Uses pydantic-settings to load configuration from environment variables.
This allows easy swapping between Gemini, Azure OpenAI, or local models.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Why pydantic-settings?
    - Type validation at startup (fail fast if config is wrong)
    - Easy to swap providers by changing env vars
    - Secrets stay in .env, not in code
    """
    
    # --- LLM Configuration ---
    # We use Gemini as the primary LLM provider
    # Gemini 2.0 Flash: Fast, 1M context, good for RAG synthesis
    # Gemini 1.5 Pro: Complex reasoning when needed
    llm_provider: Literal["gemini", "openai", "ollama"] = Field(
        default="gemini",
        description="Which LLM provider to use"
    )
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key"
    )
    gemini_model: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model to use (gemini-1.5-flash or gemini-1.5-pro)"
    )
    gemini_embedding_model: str = Field(
        default="embedding-001",
        description="Gemini embedding model (768 dimensions). Use embedding-001 for older API."
    )
    
    # --- RAG Configuration ---
    # ChromaDB for vector storage (local, no external dependencies)
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="Directory to persist ChromaDB data"
    )
    chunk_size: int = Field(
        default=500,
        description="Target chunk size in characters for semantic chunking"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks to preserve context"
    )
    retrieval_top_k: int = Field(
        default=5,
        description="Number of chunks to retrieve for RAG"
    )
    
    # --- Web Search (Tavily) ---
    tavily_api_key: str = Field(
        default="",
        description="Tavily API key for web search"
    )
    
    # --- Observability (Langfuse) ---
    # Langfuse tracks every LLM call: inputs, outputs, latency, tokens
    # This lets you debug, evaluate, and improve your AI system
    langfuse_public_key: str = Field(
        default="",
        description="Langfuse public key for tracing"
    )
    langfuse_secret_key: str = Field(
        default="",
        description="Langfuse secret key for tracing"
    )
    # Note: matches LANGFUSE_BASE_URL in .env (pydantic-settings handles the mapping)
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        alias="LANGFUSE_BASE_URL",
        description="Langfuse host URL"
    )
    
    # --- Application Settings ---
    debug: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging"
    )
    
    class Config:
        # Load from .env file if it exists
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Allow both field name and alias for env vars
        populate_by_name = True


# Singleton instance - import this in other modules
# Why singleton? We only need one config object, and it's loaded once at startup
settings = Settings()

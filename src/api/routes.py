"""
FastAPI routes for the Product Success Tracking Agent.

Exposes the pipeline and individual components via REST API.

Why FastAPI?
- Automatic OpenAPI documentation
- Pydantic integration for request/response validation
- Async support for non-blocking operations
- Easy to test and deploy
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from src.core.models import (
    PipelineInput,
    PipelineOutput,
    FeatureContextModel,
    SuccessFrameworkModel,
    ScoredTableModel,
    GrainResultModel,
    TableMetadataModel,
    ErrorDetail
)
from src.orchestrator.pipeline import ProductSuccessPipeline, run_pipeline
from src.rag.indexer import index_knowledge_base
from src.deterministic.database_explorer import explore_database, find_eligible_tables
from src.deterministic.grain_detector import detect_grain_from_model
from src.core.observability import init_observability, get_logger
from pydantic import BaseModel, Field

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Product Success Tracking Agent",
    description="RAG-powered analytics for SaaS feature success measurement",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize observability on startup
@app.on_event("startup")
async def startup_event():
    """Initialize observability and other startup tasks."""
    init_observability()
    logger.info("api_started", version="0.1.0")


# --- Request/Response Models ---

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "0.1.0"


class IndexRequest(BaseModel):
    """Request to index knowledge base."""
    knowledge_base_path: str = Field(
        default="./knowledge_base",
        description="Path to knowledge base directory"
    )


class IndexResponse(BaseModel):
    """Response from indexing operation."""
    files_processed: int
    total_chunks: int
    errors: list[str] = Field(default_factory=list)


class ExploreRequest(BaseModel):
    """Request to explore database."""
    data_directory: str = Field(
        default="./data/mock_warehouse",
        description="Path to data directory"
    )


class EligibleTablesRequest(BaseModel):
    """Request to find eligible tables."""
    keywords: list[str] = Field(description="Keywords to match against tables")
    data_directory: str = Field(
        default="./data/mock_warehouse",
        description="Path to data directory"
    )
    min_score: float = Field(default=0.3, description="Minimum relevance score")


class GrainDetectionRequest(BaseModel):
    """Request to detect grain for a table."""
    table: TableMetadataModel


# --- Health Check ---

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the service status and version.
    """
    return HealthResponse()


# --- Pipeline Endpoints ---

@app.post("/pipeline/run", response_model=PipelineOutput, tags=["Pipeline"])
async def run_full_pipeline(request: PipelineInput):
    """
    Run the complete Product Success Tracking pipeline.
    
    This endpoint:
    1. Discovers feature context via RAG
    2. Generates success framework with recommended metrics
    3. Finds eligible tables in the data warehouse
    4. Detects grain for each table
    
    Returns structured output with all analysis.
    """
    try:
        logger.info("pipeline_request", feature=request.feature_name)
        
        pipeline = ProductSuccessPipeline()
        result = pipeline.run(request)
        
        return result
    except Exception as e:
        logger.error("pipeline_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                code="PIPELINE_ERROR",
                message=str(e),
                component="pipeline",
                recoverable=False
            ).model_dump()
        )


# --- RAG Endpoints ---

@app.post("/rag/index", response_model=IndexResponse, tags=["RAG"])
async def index_documents(request: IndexRequest):
    """
    Index documents in the knowledge base.
    
    Call this endpoint to:
    - Index new documents
    - Re-index after document changes
    - Initialize the RAG system
    """
    try:
        logger.info("index_request", path=request.knowledge_base_path)
        
        stats = index_knowledge_base(request.knowledge_base_path)
        
        return IndexResponse(
            files_processed=stats.get("files_processed", 0),
            total_chunks=stats.get("total_chunks", 0),
            errors=[str(e) for e in stats.get("errors", [])]
        )
    except Exception as e:
        logger.error("index_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                code="INDEX_ERROR",
                message=str(e),
                component="rag_indexer",
                recoverable=True
            ).model_dump()
        )


# --- Database Explorer Endpoints ---

@app.post("/database/explore", response_model=list[TableMetadataModel], tags=["Database"])
async def explore_tables(request: ExploreRequest):
    """
    Explore all tables in the data directory.
    
    Returns metadata for each CSV file including:
    - Column names and types
    - Cardinality and null percentages
    - Semantic type inference
    """
    try:
        logger.info("explore_request", path=request.data_directory)
        
        tables = explore_database(request.data_directory)
        
        return tables
    except Exception as e:
        logger.error("explore_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                code="EXPLORE_ERROR",
                message=str(e),
                component="database_explorer",
                recoverable=True
            ).model_dump()
        )


@app.post("/database/eligible", response_model=list[ScoredTableModel], tags=["Database"])
async def find_relevant_tables(request: EligibleTablesRequest):
    """
    Find tables relevant to given keywords.
    
    Scores tables based on:
    - Column name match to keywords (40%)
    - Required columns present (30%)
    - Data freshness (15%)
    - Row count (15%)
    """
    try:
        logger.info("eligible_request", keywords=request.keywords)
        
        tables = find_eligible_tables(
            feature_keywords=request.keywords,
            data_directory=request.data_directory
        )
        
        # Filter by min_score
        filtered = [t for t in tables if t.score >= request.min_score]
        
        return filtered
    except Exception as e:
        logger.error("eligible_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                code="ELIGIBLE_ERROR",
                message=str(e),
                component="database_explorer",
                recoverable=True
            ).model_dump()
        )


# --- Grain Detection Endpoints ---

@app.post("/grain/detect", response_model=GrainResultModel, tags=["Grain"])
async def detect_table_grain(request: GrainDetectionRequest):
    """
    Detect the granularity of a table.
    
    Analyzes column patterns and cardinality to determine:
    - Primary grain (event, session, user, firm, time)
    - Grain column
    - Confidence score
    - Secondary grains present
    """
    try:
        logger.info("grain_request", table=request.table.name)
        
        result = detect_grain_from_model(request.table)
        
        return result
    except Exception as e:
        logger.error("grain_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                code="GRAIN_ERROR",
                message=str(e),
                component="grain_detector",
                recoverable=True
            ).model_dump()
        )


# --- Main entry point ---

def create_app() -> FastAPI:
    """
    Factory function to create the FastAPI app.
    
    Useful for testing and ASGI servers.
    """
    return app

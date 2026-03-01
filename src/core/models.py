"""
Shared Pydantic models for structured inputs/outputs across all components.

Why centralized models?
- Ensures consistent data structures between agents
- Type safety at runtime with Pydantic validation
- Clear contracts between pipeline stages
- Easy serialization to JSON for API responses
"""

from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from enum import Enum


# --- Enums ---

class GrainType(str, Enum):
    """Possible data grain levels."""
    EVENT = "event"
    SESSION = "session"
    USER = "user"
    FIRM = "firm"
    TIME = "time"
    UNKNOWN = "unknown"


class SemanticType(str, Enum):
    """Semantic types for columns."""
    IDENTIFIER = "identifier"
    TIMESTAMP = "timestamp"
    METRIC = "metric"
    DIMENSION = "dimension"


# --- RAG Models ---

class ChunkModel(BaseModel):
    """A single chunk from document processing."""
    content: str
    document_id: str
    document_name: str
    section_title: Optional[str] = None
    chunk_index: int
    start_char: int
    end_char: int


class RetrievalResultModel(BaseModel):
    """Result from hybrid retrieval."""
    content: str
    metadata: dict
    score: float
    source: str  # "semantic", "keyword", or "hybrid"


# --- Database Explorer Models ---

class ColumnInfoModel(BaseModel):
    """Metadata about a single column."""
    name: str
    dtype: str
    cardinality: int
    null_count: int
    null_pct: float
    sample_values: list[Any]
    semantic_type: SemanticType


class TableMetadataModel(BaseModel):
    """Complete metadata for a table."""
    name: str
    path: str
    columns: list[ColumnInfoModel]
    row_count: int
    date_columns: list[str] = Field(default_factory=list)
    id_columns: list[str] = Field(default_factory=list)


class ScoredTableModel(BaseModel):
    """Table with relevance score."""
    table: TableMetadataModel
    score: float
    

# --- Grain Detector Models ---

class GrainResultModel(BaseModel):
    """Result of grain detection."""
    primary_grain: GrainType
    grain_column: str
    confidence: float
    secondary_grains: list[GrainType] = Field(default_factory=list)
    reasoning: str


# --- Agent Output Models ---

class FeatureContextModel(BaseModel):
    """Structured output from context discovery agent."""
    name: str = Field(description="Feature name as identified")
    description: str = Field(description="Brief description of what the feature does")
    purpose: str = Field(description="The problem this feature solves")
    target_users: list[str] = Field(default_factory=list, description="Who uses this feature")
    success_criteria: list[str] = Field(default_factory=list, description="How success was defined")
    recommended_metrics: list[str] = Field(default_factory=list, description="KPIs to track")
    related_features: list[str] = Field(default_factory=list, description="Related features")
    industry_benchmarks: dict = Field(default_factory=dict, description="Industry benchmarks")
    confidence: float = Field(default=0.0, description="Confidence score 0-1")
    sources: list[str] = Field(default_factory=list, description="Source documents")


class MetricDefinitionModel(BaseModel):
    """Definition of a single metric."""
    name: str
    display_name: str
    description: str
    formula: str
    target: Optional[str] = None
    frequency: str


class TrackingEventModel(BaseModel):
    """An event to track."""
    event_name: str
    trigger: str
    properties: list[str] = Field(default_factory=list)


class SuccessFrameworkModel(BaseModel):
    """Complete success measurement framework."""
    feature_name: str
    primary_metrics: list[MetricDefinitionModel] = Field(default_factory=list)
    secondary_metrics: list[MetricDefinitionModel] = Field(default_factory=list)
    tracking_events: list[TrackingEventModel] = Field(default_factory=list)
    analysis_approach: str = ""
    data_requirements: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


# --- Pipeline Models ---

class PipelineInput(BaseModel):
    """Input to the pipeline."""
    feature_name: str = Field(description="Name of the feature to analyze")
    additional_context: str = Field(default="", description="Optional additional context")
    skip_indexing: bool = Field(default=False, description="Skip re-indexing knowledge base")


class PipelineOutput(BaseModel):
    """Complete output from the pipeline."""
    feature_context: FeatureContextModel
    success_framework: SuccessFrameworkModel
    eligible_tables: list[ScoredTableModel] = Field(default_factory=list)
    grain_results: dict[str, GrainResultModel] = Field(default_factory=dict)
    llm_calls: int = 0
    errors: list[str] = Field(default_factory=list)
    execution_time_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# --- Error Models ---

class ErrorDetail(BaseModel):
    """Structured error information."""
    code: str
    message: str
    component: str
    recoverable: bool = True
    details: Optional[dict] = None


class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    
    def __init__(self, error: ErrorDetail):
        self.error = error
        super().__init__(error.message)

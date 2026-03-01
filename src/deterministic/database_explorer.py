"""
Database Explorer - Deterministic schema introspection.

Extracts metadata from CSV files (simulating a data warehouse) without using LLM.
This is pure pattern matching and SQL-like operations.

Why no LLM?
- Schema introspection is pattern matching, not reasoning
- Column names follow conventions (user_id, created_at, etc.)
- Data types are explicit
- LLM would just be expensive pattern matching
"""

import pandas as pd
from typing import Optional, Any
from pathlib import Path
from rapidfuzz import fuzz
from src.core.observability import get_logger
from src.core.models import (
    ColumnInfoModel,
    TableMetadataModel,
    ScoredTableModel,
    SemanticType,
    ErrorDetail,
    PipelineError
)

logger = get_logger(__name__)


# Keep dataclasses for internal use, convert to Pydantic for output
class ColumnInfo:
    """
    Internal representation of column metadata.
    """
    def __init__(
        self,
        name: str,
        dtype: str,
        cardinality: int,
        null_count: int,
        null_pct: float,
        sample_values: list[Any],
        semantic_type: str
    ):
        self.name = name
        self.dtype = dtype
        self.cardinality = cardinality
        self.null_count = null_count
        self.null_pct = null_pct
        self.sample_values = sample_values
        self.semantic_type = semantic_type
    
    def to_model(self) -> ColumnInfoModel:
        """Convert to Pydantic model for structured output."""
        return ColumnInfoModel(
            name=self.name,
            dtype=self.dtype,
            cardinality=self.cardinality,
            null_count=self.null_count,
            null_pct=self.null_pct,
            sample_values=self.sample_values,
            semantic_type=SemanticType(self.semantic_type)
        )


class TableMetadata:
    """
    Internal representation of table metadata.
    """
    def __init__(
        self,
        name: str,
        path: str,
        columns: list[ColumnInfo],
        row_count: int,
        date_columns: list[str] = None,
        id_columns: list[str] = None
    ):
        self.name = name
        self.path = path
        self.columns = columns
        self.row_count = row_count
        self.date_columns = date_columns or []
        self.id_columns = id_columns or []
    
    def to_model(self) -> TableMetadataModel:
        """Convert to Pydantic model for structured output."""
        return TableMetadataModel(
            name=self.name,
            path=self.path,
            columns=[c.to_model() for c in self.columns],
            row_count=self.row_count,
            date_columns=self.date_columns,
            id_columns=self.id_columns
        )
    

class DatabaseExplorer:
    """
    Explores CSV files to extract schema metadata.
    
    Simulates data warehouse exploration by:
    1. Reading CSV files from a directory
    2. Extracting column metadata (types, cardinality, samples)
    3. Inferring semantic types (identifier, timestamp, metric, dimension)
    """
    
    # Patterns for semantic type detection
    ID_PATTERNS = ['_id', 'id_', '_key', 'key_', '_pk', 'uuid', 'guid']
    TIMESTAMP_PATTERNS = ['timestamp', 'created', 'updated', 'date', 'time', '_at']
    METRIC_PATTERNS = ['count', 'sum', 'total', 'amount', 'value', 'rate', 'pct', 'percent']
    
    def __init__(self, data_directory: str = "./data/mock_warehouse"):
        """
        Initialize the explorer.
        
        Args:
            data_directory: Path to directory containing CSV files
        """
        self.data_directory = Path(data_directory)
    
    def _infer_semantic_type(self, col_name: str, dtype: str, sample_values: list) -> str:
        """
        Infer the semantic type of a column.
        
        Types:
        - identifier: Primary/foreign keys (user_id, session_id)
        - timestamp: Date/time columns
        - metric: Numeric measures (counts, amounts)
        - dimension: Categorical attributes
        """
        col_lower = col_name.lower()
        
        # Check for identifier patterns
        for pattern in self.ID_PATTERNS:
            if pattern in col_lower:
                return "identifier"
        
        # Check for timestamp patterns
        for pattern in self.TIMESTAMP_PATTERNS:
            if pattern in col_lower:
                return "timestamp"
        
        # Check for metric patterns
        for pattern in self.METRIC_PATTERNS:
            if pattern in col_lower:
                return "metric"
        
        # Infer from dtype
        if 'int' in dtype or 'float' in dtype:
            return "metric"
        
        # Default to dimension
        return "dimension"
    
    def explore_table(self, file_path: Path) -> TableMetadata:
        """
        Extract metadata from a single CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            TableMetadata with column info
            
        Raises:
            PipelineError: If file cannot be read or parsed
        """
        # Validate file exists
        if not file_path.exists():
            raise PipelineError(ErrorDetail(
                code="FILE_NOT_FOUND",
                message=f"CSV file not found: {file_path}",
                component="database_explorer",
                recoverable=False
            ))
        
        # Read CSV with error handling
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            raise PipelineError(ErrorDetail(
                code="EMPTY_FILE",
                message=f"CSV file is empty: {file_path}",
                component="database_explorer",
                recoverable=True
            ))
        except Exception as e:
            raise PipelineError(ErrorDetail(
                code="CSV_PARSE_ERROR",
                message=f"Failed to parse CSV: {file_path}",
                component="database_explorer",
                recoverable=False,
                details={"error": str(e)}
            ))
        
        columns = []
        date_columns = []
        id_columns = []
        
        for col in df.columns:
            # Get basic stats
            cardinality = df[col].nunique()
            null_count = df[col].isna().sum()
            null_pct = null_count / len(df) if len(df) > 0 else 0
            
            # Get sample values (non-null, unique)
            samples = df[col].dropna().unique()[:5].tolist()
            
            # Get dtype as string
            dtype = str(df[col].dtype)
            
            # Infer semantic type
            semantic_type = self._infer_semantic_type(col, dtype, samples)
            
            col_info = ColumnInfo(
                name=col,
                dtype=dtype,
                cardinality=cardinality,
                null_count=int(null_count),
                null_pct=float(null_pct),
                sample_values=samples,
                semantic_type=semantic_type
            )
            columns.append(col_info)
            
            # Track special columns
            if semantic_type == "timestamp":
                date_columns.append(col)
            elif semantic_type == "identifier":
                id_columns.append(col)
        
        return TableMetadata(
            name=file_path.stem,
            path=str(file_path),
            columns=columns,
            row_count=len(df),
            date_columns=date_columns,
            id_columns=id_columns
        )
    
    def explore_all(self) -> list[TableMetadata]:
        """
        Explore all CSV files in the data directory.
        
        Returns:
            List of TableMetadata for each file
            
        Raises:
            PipelineError: If directory doesn't exist
        """
        # Validate directory exists
        if not self.data_directory.exists():
            raise PipelineError(ErrorDetail(
                code="DIRECTORY_NOT_FOUND",
                message=f"Data directory not found: {self.data_directory}",
                component="database_explorer",
                recoverable=False
            ))
        
        tables = []
        errors = []
        
        csv_files = list(self.data_directory.glob("*.csv"))
        if not csv_files:
            logger.warning("no_csv_files_found", directory=str(self.data_directory))
            return tables
        
        for csv_file in csv_files:
            try:
                metadata = self.explore_table(csv_file)
                tables.append(metadata)
                logger.info("explored_table", table=metadata.name, columns=len(metadata.columns))
            except PipelineError as e:
                # Log but continue with other files
                errors.append(e.error)
                logger.error("explore_error", file=str(csv_file), error=e.error.message)
            except Exception as e:
                # Unexpected error - log and continue
                logger.error("explore_error", file=str(csv_file), error=str(e))
        
        if errors:
            logger.warning("exploration_completed_with_errors", 
                          tables_found=len(tables), 
                          errors=len(errors))
        
        return tables
    
    def score_table_relevance(
        self,
        table: TableMetadata,
        feature_keywords: list[str],
        required_columns: list[str] = None
    ) -> float:
        """
        Score how relevant a table is for analyzing a feature.
        
        Scoring (0-1):
        - 40%: Column name match to feature keywords
        - 30%: Has required columns (user_id, timestamp)
        - 15%: Data freshness (has recent timestamps)
        - 15%: Row count (sufficient data)
        
        Args:
            table: TableMetadata to score
            feature_keywords: Keywords related to the feature
            required_columns: Columns that must be present
            
        Returns:
            Relevance score 0-1
        """
        required = required_columns or ["user_id", "timestamp"]
        
        # Score 1: Column name match (40%)
        column_names = [c.name.lower() for c in table.columns]
        keyword_matches = 0
        for keyword in feature_keywords:
            for col in column_names:
                if fuzz.partial_ratio(keyword.lower(), col) > 70:
                    keyword_matches += 1
                    break
        
        keyword_score = min(keyword_matches / max(len(feature_keywords), 1), 1.0)
        
        # Score 2: Required columns (30%)
        required_found = 0
        for req in required:
            for col in column_names:
                if fuzz.partial_ratio(req.lower(), col) > 80:
                    required_found += 1
                    break
        
        required_score = required_found / max(len(required), 1)
        
        # Score 3: Has timestamp columns (15%)
        timestamp_score = 1.0 if table.date_columns else 0.0
        
        # Score 4: Row count (15%)
        # Assume 100+ rows is sufficient
        row_score = min(table.row_count / 100, 1.0)
        
        # Weighted combination
        total_score = (
            0.40 * keyword_score +
            0.30 * required_score +
            0.15 * timestamp_score +
            0.15 * row_score
        )
        
        return round(total_score, 3)
    
    def find_eligible_tables(
        self,
        feature_keywords: list[str],
        min_score: float = 0.3
    ) -> list[ScoredTableModel]:
        """
        Find tables relevant to a feature.
        
        Args:
            feature_keywords: Keywords from feature context
            min_score: Minimum relevance score to include
            
        Returns:
            List of ScoredTableModel (Pydantic), sorted by score
        """
        # Validate input
        if not feature_keywords:
            logger.warning("no_keywords_provided")
            feature_keywords = []
        
        tables = self.explore_all()
        
        scored = []
        for table in tables:
            score = self.score_table_relevance(table, feature_keywords)
            if score >= min_score:
                # Convert to Pydantic model for structured output
                scored.append(ScoredTableModel(
                    table=table.to_model(),
                    score=score
                ))
        
        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)
        
        return scored


def explore_database(data_directory: str = "./data/mock_warehouse") -> list[TableMetadataModel]:
    """
    Convenience function to explore all tables.
    
    Returns:
        List of TableMetadataModel (Pydantic) for structured output
    """
    explorer = DatabaseExplorer(data_directory)
    tables = explorer.explore_all()
    return [t.to_model() for t in tables]


def find_eligible_tables(
    feature_keywords: list[str],
    data_directory: str = "./data/mock_warehouse"
) -> list[ScoredTableModel]:
    """
    Convenience function to find relevant tables.
    
    Returns:
        List of ScoredTableModel (Pydantic) for structured output
    """
    explorer = DatabaseExplorer(data_directory)
    return explorer.find_eligible_tables(feature_keywords)

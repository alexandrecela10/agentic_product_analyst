"""
Grain Detector - Deterministic data granularity detection.

Detects the level of detail in a dataset:
- Event level (one row per action)
- Session level (one row per visit)
- User level (one row per user)
- Firm level (one row per company)
- Time level (aggregated by time period)

Why no LLM?
- Grain patterns are finite and known
- Detection uses cardinality math and pattern matching
- Column naming conventions are consistent
"""

from typing import Optional
from src.deterministic.database_explorer import TableMetadata, ColumnInfo
from src.core.observability import get_logger
from src.core.models import (
    GrainResultModel,
    GrainType,
    TableMetadataModel,
    ErrorDetail,
    PipelineError
)

logger = get_logger(__name__)


# Grain hierarchy with column patterns and indicators
GRAIN_HIERARCHY = {
    "event": {
        "columns": ["event_id", "action_id", "log_id", "click_id", "conversion_id"],
        "priority": 1,
        "description": "One row per user action/event"
    },
    "session": {
        "columns": ["session_id", "visit_id"],
        "priority": 2,
        "description": "One row per user session/visit"
    },
    "user": {
        "columns": ["user_id", "customer_id", "account_id", "member_id", "anonymous_id"],
        "priority": 3,
        "description": "One row per user"
    },
    "firm": {
        "columns": ["company_id", "org_id", "tenant_id", "workspace_id", "firm_id"],
        "priority": 4,
        "description": "One row per company/organization"
    },
    "time": {
        "columns": ["date", "week", "month", "quarter", "year"],
        "priority": 5,
        "description": "Aggregated by time period"
    }
}


class GrainDetector:
    """
    Detects the granularity of a dataset.
    
    Algorithm:
    1. Find all identifier columns
    2. Check cardinality ratios (unique values vs row count)
    3. Match column names to grain hierarchy
    4. Determine primary grain based on highest cardinality ratio
    """
    
    def __init__(self):
        self.grain_hierarchy = GRAIN_HIERARCHY
    
    def _find_grain_columns(self, table: TableMetadata) -> dict[str, list[str]]:
        """
        Find columns that match each grain type.
        
        Returns:
            Dict mapping grain type to matching column names
        """
        matches = {grain: [] for grain in self.grain_hierarchy}
        
        for col in table.columns:
            col_lower = col.name.lower()
            
            for grain, config in self.grain_hierarchy.items():
                for pattern in config["columns"]:
                    if pattern in col_lower or col_lower == pattern:
                        matches[grain].append(col.name)
                        break
        
        return matches
    
    def _calculate_cardinality_ratio(
        self,
        column: ColumnInfo,
        row_count: int
    ) -> float:
        """
        Calculate cardinality ratio (unique values / total rows).
        
        Ratio close to 1.0 = likely primary grain
        Ratio close to 0.0 = likely dimension/attribute
        """
        if row_count == 0:
            return 0.0
        return column.cardinality / row_count
    
    def detect(self, table: TableMetadata) -> GrainResultModel:
        """
        Detect the grain of a table.
        
        Args:
            table: TableMetadata to analyze
            
        Returns:
            GrainResultModel (Pydantic) with detected grain
            
        Raises:
            PipelineError: If table has no columns
        """
        # Validate input
        if not table.columns:
            raise PipelineError(ErrorDetail(
                code="EMPTY_TABLE",
                message=f"Table '{table.name}' has no columns",
                component="grain_detector",
                recoverable=True
            ))
        # Step 1: Find columns matching each grain type
        grain_columns = self._find_grain_columns(table)
        
        # Step 2: Calculate cardinality ratios for identifier columns
        candidates = []
        
        for col in table.columns:
            if col.semantic_type == "identifier":
                ratio = self._calculate_cardinality_ratio(col, table.row_count)
                
                # Find which grain this column belongs to
                grain_type = None
                for grain, cols in grain_columns.items():
                    if col.name in cols:
                        grain_type = grain
                        break
                
                if grain_type:
                    candidates.append({
                        "column": col.name,
                        "grain": grain_type,
                        "ratio": ratio,
                        "priority": self.grain_hierarchy[grain_type]["priority"]
                    })
        
        # Step 3: Determine primary grain
        if not candidates:
            # No identifier columns found - check for time-based grain
            if table.date_columns:
                return GrainResultModel(
                    primary_grain=GrainType.TIME,
                    grain_column=table.date_columns[0],
                    confidence=0.6,
                    secondary_grains=[],
                    reasoning="No identifier columns found; detected time-based aggregation"
                )
            else:
                return GrainResultModel(
                    primary_grain=GrainType.UNKNOWN,
                    grain_column="",
                    confidence=0.3,
                    secondary_grains=[],
                    reasoning="Could not determine grain - no identifiers or timestamps found"
                )
        
        # Sort by cardinality ratio (highest first) - highest ratio = most granular
        candidates.sort(key=lambda x: x["ratio"], reverse=True)
        
        # Primary grain is the one with highest cardinality ratio
        primary = candidates[0]
        
        # If ratio is close to 1.0, high confidence
        if primary["ratio"] > 0.95:
            confidence = 0.95
            reasoning = f"Column '{primary['column']}' has 1:1 relationship with rows (ratio: {primary['ratio']:.2f})"
        elif primary["ratio"] > 0.5:
            confidence = 0.8
            reasoning = f"Column '{primary['column']}' has high cardinality (ratio: {primary['ratio']:.2f})"
        else:
            confidence = 0.6
            reasoning = f"Column '{primary['column']}' detected but low cardinality (ratio: {primary['ratio']:.2f})"
        
        # Secondary grains are other identifier types present
        secondary = [c["grain"] for c in candidates[1:] if c["grain"] != primary["grain"]]
        secondary = list(dict.fromkeys(secondary))  # Remove duplicates, preserve order
        
        # Convert to GrainType enum
        try:
            primary_grain_type = GrainType(primary["grain"])
        except ValueError:
            primary_grain_type = GrainType.UNKNOWN
        
        secondary_grain_types = []
        for g in secondary:
            try:
                secondary_grain_types.append(GrainType(g))
            except ValueError:
                pass
        
        return GrainResultModel(
            primary_grain=primary_grain_type,
            grain_column=primary["column"],
            confidence=confidence,
            secondary_grains=secondary_grain_types,
            reasoning=reasoning
        )


def detect_grain(table: TableMetadata) -> GrainResultModel:
    """
    Convenience function to detect grain.
    
    Args:
        table: TableMetadata (internal) or TableMetadataModel (Pydantic)
        
    Returns:
        GrainResultModel (Pydantic) for structured output
    """
    detector = GrainDetector()
    return detector.detect(table)


def detect_grain_from_model(table_model: TableMetadataModel) -> GrainResultModel:
    """
    Detect grain from a Pydantic TableMetadataModel.
    
    Converts Pydantic model to internal representation for processing.
    """
    # Convert Pydantic model to internal TableMetadata
    columns = [
        ColumnInfo(
            name=c.name,
            dtype=c.dtype,
            cardinality=c.cardinality,
            null_count=c.null_count,
            null_pct=c.null_pct,
            sample_values=c.sample_values,
            semantic_type=c.semantic_type.value
        )
        for c in table_model.columns
    ]
    
    table = TableMetadata(
        name=table_model.name,
        path=table_model.path,
        columns=columns,
        row_count=table_model.row_count,
        date_columns=table_model.date_columns,
        id_columns=table_model.id_columns
    )
    
    detector = GrainDetector()
    return detector.detect(table)

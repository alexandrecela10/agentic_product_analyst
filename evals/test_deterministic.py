"""
Tests for deterministic components (no LLM calls).

Why test deterministic components?
- Fast to run (no API calls)
- Predictable outputs
- Catch regressions early
- Build confidence in the system

These tests run without API keys and complete in seconds.
"""

import pytest
from src.deterministic.database_explorer import explore_database, find_eligible_tables
from src.deterministic.grain_detector import detect_grain_from_model
from evals.datasets import DATABASE_EXPLORER_CASES, GRAIN_DETECTION_CASES


class TestDatabaseExplorer:
    """
    Test the database explorer component.
    
    What we're testing:
    - Can it find all tables in the data directory?
    - Does it extract column metadata correctly?
    - Does it score table relevance accurately?
    """
    
    def test_explore_all_tables(self):
        """Test: Should find all tables in mock warehouse."""
        tables = explore_database('./data/mock_warehouse')
        
        # Should find all 5 tables
        assert len(tables) == 5, f"Expected 5 tables, found {len(tables)}"
        
        # Check table names
        table_names = {t.name for t in tables}
        expected = {'search_events', 'search_clicks', 'conversions', 'users', 'sessions'}
        assert table_names == expected, f"Missing tables: {expected - table_names}"
    
    def test_column_metadata(self):
        """Test: Should extract column info correctly."""
        tables = explore_database('./data/mock_warehouse')
        
        # Find search_events table
        search_events = next(t for t in tables if t.name == 'search_events')
        
        # Should have expected columns (using actual column names from CSV)
        column_names = {c.name for c in search_events.columns}
        assert 'event_id' in column_names
        assert 'user_id' in column_names
        assert 'query' in column_names  # Fixed: actual column name is 'query' not 'search_query'
        assert 'timestamp' in column_names
    
    @pytest.mark.parametrize("case", DATABASE_EXPLORER_CASES)
    def test_find_eligible_tables(self, case):
        """Test: Should find relevant tables based on keywords."""
        eligible = find_eligible_tables(
            feature_keywords=case["keywords"],
            data_directory='./data/mock_warehouse'
        )
        
        # Should find minimum number of tables
        assert len(eligible) >= case["min_tables"], \
            f"Expected at least {case['min_tables']} tables, found {len(eligible)}"
        
        # Should include expected tables
        found_names = {t.table.name for t in eligible}
        for expected_table in case["expected_tables"]:
            assert expected_table in found_names, \
                f"Expected to find {expected_table}, got {found_names}"


class TestGrainDetector:
    """
    Test the grain detection component.
    
    What we're testing:
    - Does it correctly identify event-level tables?
    - Does it correctly identify user-level tables?
    - Does it correctly identify session-level tables?
    - Is the confidence score reasonable?
    """
    
    @pytest.mark.parametrize("case", GRAIN_DETECTION_CASES)
    def test_grain_detection(self, case):
        """Test: Should detect correct grain for each table."""
        # First get the table metadata
        tables = explore_database('./data/mock_warehouse')
        table = next(t for t in tables if t.name == case["table_name"])
        
        # Detect grain
        result = detect_grain_from_model(table)
        
        # Check grain type
        assert result.primary_grain.value == case["expected_grain"], \
            f"Expected {case['expected_grain']}, got {result.primary_grain.value}"
        
        # Check confidence
        assert result.confidence >= case["min_confidence"], \
            f"Confidence {result.confidence} below threshold {case['min_confidence']}"
    
    def test_grain_reasoning(self):
        """Test: Should provide reasoning for grain detection."""
        tables = explore_database('./data/mock_warehouse')
        search_events = next(t for t in tables if t.name == 'search_events')
        
        result = detect_grain_from_model(search_events)
        
        # Should have reasoning
        assert result.reasoning, "Should provide reasoning"
        assert len(result.reasoning) > 0, "Reasoning should not be empty"
        
        # Reasoning should mention the grain column
        assert result.grain_column in result.reasoning, \
            f"Reasoning should mention grain column {result.grain_column}"


class TestDataQuality:
    """
    Test data quality checks.
    
    What we're testing:
    - Are there any null values in key columns?
    - Are cardinalities reasonable?
    - Are semantic types inferred correctly?
    """
    
    def test_low_nulls_in_primary_ids(self):
        """Test: Primary ID columns should have low null rates."""
        tables = explore_database('./data/mock_warehouse')
        
        for table in tables:
            for column in table.columns:
                # Primary IDs (event_id, session_id, etc.) should have no nulls
                if column.name.endswith('_id') and column.name.startswith(table.name.rstrip('s')):
                    assert column.null_pct == 0.0, \
                        f"Primary key {table.name}.{column.name} has {column.null_pct}% nulls"
                # Foreign keys can have some nulls (e.g., anonymous users)
                elif 'id' in column.name.lower():
                    assert column.null_pct < 50.0, \
                        f"{table.name}.{column.name} has excessive nulls: {column.null_pct}%"
    
    def test_semantic_types(self):
        """Test: Should infer semantic types correctly."""
        tables = explore_database('./data/mock_warehouse')
        search_events = next(t for t in tables if t.name == 'search_events')
        
        # Check semantic types
        semantic_types = {c.name: c.semantic_type for c in search_events.columns}
        
        assert semantic_types.get('event_id') == 'identifier'
        assert semantic_types.get('user_id') == 'identifier'
        assert semantic_types.get('timestamp') == 'timestamp'


# Run with: uv run pytest evals/test_deterministic.py -v

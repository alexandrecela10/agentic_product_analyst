"""
End-to-end pipeline tests.

Why test the full pipeline?
- Validates all components work together
- Catches integration issues
- Measures real-world performance
- Ensures output quality

These tests run the complete pipeline and are the most important.
"""

import pytest
from src.orchestrator.pipeline import run_pipeline
from src.core.observability import init_observability
from evals.datasets import PIPELINE_CASES, GOLDEN_OUTPUTS

# Initialize observability
init_observability()


class TestPipelineExecution:
    """
    Test full pipeline execution.
    
    What we're testing:
    - Does the pipeline complete without errors?
    - Does it produce all expected outputs?
    - Is it fast enough?
    - Are the outputs high quality?
    """
    
    @pytest.mark.parametrize("case", PIPELINE_CASES)
    def test_pipeline_completes(self, case):
        """Test: Pipeline should complete successfully."""
        result = run_pipeline(
            feature_name=case["feature_name"],
            additional_context=case["additional_context"]
        )
        
        # Should complete without errors
        assert len(result.errors) == 0, f"Pipeline had errors: {result.errors}"
        
        # Should have all components
        assert result.feature_context is not None, "Missing feature context"
        assert result.success_framework is not None, "Missing success framework"
        assert len(result.eligible_tables) > 0, "No eligible tables found"
        assert len(result.grain_results) > 0, "No grain results"
    
    @pytest.mark.parametrize("case", PIPELINE_CASES)
    def test_pipeline_performance(self, case):
        """Test: Pipeline should complete within time limit."""
        result = run_pipeline(
            feature_name=case["feature_name"],
            additional_context=case["additional_context"]
        )
        
        assert result.execution_time_ms <= case["max_execution_time_ms"], \
            f"Pipeline took {result.execution_time_ms}ms, limit is {case['max_execution_time_ms']}ms"
    
    @pytest.mark.parametrize("case", PIPELINE_CASES)
    def test_pipeline_finds_expected_tables(self, case):
        """Test: Should find expected tables."""
        result = run_pipeline(
            feature_name=case["feature_name"],
            additional_context=case["additional_context"]
        )
        
        found_tables = {t.table.name for t in result.eligible_tables}
        
        for expected_table in case["expected_eligible_tables"]:
            assert expected_table in found_tables, \
                f"Expected to find {expected_table}, got {found_tables}"
    
    @pytest.mark.parametrize("case", PIPELINE_CASES)
    def test_pipeline_detects_correct_grains(self, case):
        """Test: Should detect correct grain for each table."""
        result = run_pipeline(
            feature_name=case["feature_name"],
            additional_context=case["additional_context"]
        )
        
        for table_name, expected_grain in case["expected_grains"].items():
            assert table_name in result.grain_results, \
                f"Missing grain result for {table_name}"
            
            actual_grain = result.grain_results[table_name].primary_grain.value
            assert actual_grain == expected_grain, \
                f"Expected {expected_grain} for {table_name}, got {actual_grain}"


class TestPipelineOutputQuality:
    """
    Test the quality of pipeline outputs.
    
    What we're testing:
    - Are metrics relevant to the feature?
    - Is the context accurate?
    - Are tables properly scored?
    """
    
    def test_context_has_high_confidence(self):
        """Test: Context discovery should be confident."""
        result = run_pipeline("AI Search")
        
        assert result.feature_context.confidence >= 0.6, \
            f"Low confidence: {result.feature_context.confidence}"
    
    def test_metrics_are_specific(self):
        """Test: Metrics should have detailed descriptions."""
        result = run_pipeline("AI Search")
        
        for metric in result.success_framework.primary_metrics:
            # Description should be substantial
            assert len(metric.description) > 20, \
                f"Metric {metric.display_name} has short description"
            
            # Should have calculation method
            assert len(metric.calculation) > 10, \
                f"Metric {metric.display_name} has short calculation"
    
    def test_tables_are_scored(self):
        """Test: All tables should have relevance scores."""
        result = run_pipeline("AI Search")
        
        for scored_table in result.eligible_tables:
            assert 0 <= scored_table.score <= 1.0, \
                f"Invalid score {scored_table.score} for {scored_table.table.name}"


class TestGoldenExamples:
    """
    Test against known good outputs (golden examples).
    
    What we're testing:
    - Does output match expected metrics?
    - Is there consistency across runs?
    - Are we maintaining quality?
    
    This is regression testing - if these fail, something changed.
    """
    
    def test_ai_search_metrics_match_golden(self):
        """Test: AI Search should produce expected metrics."""
        result = run_pipeline("AI Search")
        
        golden = GOLDEN_OUTPUTS["AI Search"]
        actual_primary = [m.display_name for m in result.success_framework.primary_metrics]
        
        # Check for overlap with golden metrics
        # We don't require exact match (LLM can vary) but should have some overlap
        overlap = set(actual_primary) & set(golden["primary_metrics"])
        
        assert len(overlap) >= 2, \
            f"Expected at least 2 matching metrics, got {overlap}. Actual: {actual_primary}"
    
    def test_consistency_across_runs(self):
        """Test: Multiple runs should produce similar results."""
        # Run pipeline twice
        result1 = run_pipeline("AI Search")
        result2 = run_pipeline("AI Search")
        
        # Should find same tables
        tables1 = {t.table.name for t in result1.eligible_tables}
        tables2 = {t.table.name for t in result2.eligible_tables}
        
        assert tables1 == tables2, \
            f"Inconsistent table discovery: {tables1} vs {tables2}"
        
        # Should detect same grains
        grains1 = {name: g.primary_grain.value for name, g in result1.grain_results.items()}
        grains2 = {name: g.primary_grain.value for name, g in result2.grain_results.items()}
        
        assert grains1 == grains2, \
            f"Inconsistent grain detection: {grains1} vs {grains2}"


class TestErrorHandling:
    """
    Test error handling and edge cases.
    
    What we're testing:
    - Does it handle missing data gracefully?
    - Does it handle invalid inputs?
    - Are errors properly logged?
    """
    
    def test_handles_unknown_feature(self):
        """Test: Should handle unknown features gracefully."""
        result = run_pipeline("Completely Unknown Feature XYZ123")
        
        # Should still complete (maybe with lower confidence)
        assert result.feature_context is not None
        
        # Confidence might be lower
        # But should not crash
    
    def test_handles_empty_additional_context(self):
        """Test: Should work with empty additional context."""
        result = run_pipeline("AI Search", additional_context="")
        
        assert len(result.errors) == 0
        assert result.feature_context is not None


# Run with different pytest options:
# 
# Fast tests only:
#   pytest evals/test_pipeline.py -m "not slow"
#
# All tests with verbose output:
#   pytest evals/test_pipeline.py -v
#
# Stop on first failure:
#   pytest evals/test_pipeline.py -x
#
# Show print statements:
#   pytest evals/test_pipeline.py -s

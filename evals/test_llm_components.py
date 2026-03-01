"""
Tests for LLM-powered components.

Why test LLM components?
- Ensure prompts produce expected outputs
- Catch prompt regressions
- Validate structured output parsing
- Monitor quality over time

These tests require API keys and make real LLM calls.
They're slower but essential for quality assurance.
"""

import pytest
from src.agents.context_discovery import ContextDiscoveryAgent
from src.agents.success_framework import SuccessFrameworkAgent
from src.core.observability import init_observability, get_langfuse
from evals.datasets import (
    CONTEXT_DISCOVERY_CASES,
    SUCCESS_FRAMEWORK_CASES,
    JUDGE_PROMPT_CONTEXT_QUALITY,
    JUDGE_PROMPT_METRICS_QUALITY
)
from src.core.llm_client import generate_text_sync

# Initialize observability for all tests
init_observability()


class TestContextDiscoveryAgent:
    """
    Test the context discovery agent.
    
    What we're testing:
    - Does it extract relevant information from docs?
    - Is the confidence score reasonable?
    - Does it identify target users correctly?
    - Does the output match expected keywords?
    """
    
    @pytest.mark.parametrize("case", CONTEXT_DISCOVERY_CASES)
    def test_context_discovery_keywords(self, case):
        """Test: Should find expected keywords in context."""
        agent = ContextDiscoveryAgent()
        
        result = agent.discover(
            feature_name=case["feature_name"],
            additional_context=case["additional_context"]
        )
        
        # Combine all text fields for keyword search
        all_text = f"{result.description} {result.purpose} {result.target_users}".lower()
        
        # Check for expected keywords
        found_keywords = [kw for kw in case["expected_keywords"] if kw in all_text]
        
        assert len(found_keywords) >= 2, \
            f"Expected at least 2 keywords from {case['expected_keywords']}, found {found_keywords}"
    
    @pytest.mark.parametrize("case", CONTEXT_DISCOVERY_CASES)
    def test_context_discovery_confidence(self, case):
        """Test: Should have reasonable confidence score."""
        agent = ContextDiscoveryAgent()
        
        result = agent.discover(
            feature_name=case["feature_name"],
            additional_context=case["additional_context"]
        )
        
        assert result.confidence >= case["min_confidence"], \
            f"Confidence {result.confidence} below threshold {case['min_confidence']}"
        
        assert result.confidence <= 1.0, \
            f"Confidence {result.confidence} exceeds 1.0"
    
    def test_context_discovery_structure(self):
        """Test: Should return all required fields."""
        agent = ContextDiscoveryAgent()
        result = agent.discover("AI Search")
        
        # Check all fields are present and non-empty
        assert result.name, "Name should not be empty"
        assert result.description, "Description should not be empty"
        assert result.purpose, "Purpose should not be empty"
        assert result.target_users, "Target users should not be empty"
        assert len(result.success_criteria) > 0, "Should have success criteria"
        assert len(result.recommended_metrics) > 0, "Should have recommended metrics"


class TestSuccessFrameworkAgent:
    """
    Test the success framework generation agent.
    
    What we're testing:
    - Does it generate enough metrics?
    - Are metrics specific and measurable?
    - Do they align with the feature type?
    - Is the output properly structured?
    """
    
    @pytest.mark.parametrize("case", SUCCESS_FRAMEWORK_CASES)
    def test_success_framework_metric_count(self, case):
        """Test: Should generate minimum number of metrics."""
        agent = SuccessFrameworkAgent()
        
        # Create mock context
        from src.core.models import FeatureContextModel
        context = FeatureContextModel(
            name=case["feature_name"],
            description=case["feature_context"],
            purpose=case["feature_context"],
            target_users="all users",
            success_criteria=["increase engagement"],
            recommended_metrics=["usage rate"],
            confidence=0.8
        )
        
        result = agent.generate(context)
        
        # Check metric counts
        assert len(result.primary_metrics) >= case["min_primary_metrics"], \
            f"Expected at least {case['min_primary_metrics']} primary metrics, got {len(result.primary_metrics)}"
        
        assert len(result.secondary_metrics) >= case["min_secondary_metrics"], \
            f"Expected at least {case['min_secondary_metrics']} secondary metrics, got {len(result.secondary_metrics)}"
    
    def test_success_framework_metric_structure(self):
        """Test: Metrics should have all required fields."""
        agent = SuccessFrameworkAgent()
        
        from src.core.models import FeatureContextModel
        context = FeatureContextModel(
            name="AI Search",
            description="AI-powered search feature",
            purpose="Help users find content faster",
            target_users="all users",
            success_criteria=["increase search usage"],
            recommended_metrics=["search conversion rate"],
            confidence=0.8
        )
        
        result = agent.generate(context)
        
        # Check primary metrics have all fields
        for metric in result.primary_metrics:
            assert metric.display_name, "Metric should have display_name"
            assert metric.description, "Metric should have description"
            assert metric.calculation, "Metric should have calculation"
            assert metric.metric_type, "Metric should have metric_type"


class TestLLMAsJudge:
    """
    Use LLM to evaluate output quality.
    
    What we're testing:
    - Overall quality of generated content
    - Alignment with best practices
    - Usefulness for end users
    
    This is more subjective but catches quality issues.
    """
    
    @pytest.mark.slow
    def test_context_quality_with_judge(self):
        """Test: LLM judge should rate context quality highly."""
        agent = ContextDiscoveryAgent()
        result = agent.discover("AI Search")
        
        # Use LLM as judge
        judge_prompt = JUDGE_PROMPT_CONTEXT_QUALITY.format(
            feature_name=result.name,
            description=result.description,
            purpose=result.purpose,
            target_users=result.target_users
        )
        
        judge_response = generate_text_sync(
            prompt=judge_prompt,
            trace_name="llm_judge_context_quality",
            trace_metadata={"feature": result.name}
        )
        
        # Extract score (simple parsing - assumes LLM returns number)
        try:
            score = float(judge_response.strip().split('\n')[-1])
            assert score >= 7.0, f"Quality score {score} below threshold 7.0"
        except ValueError:
            pytest.skip(f"Could not parse judge score from: {judge_response}")
    
    @pytest.mark.slow
    def test_metrics_quality_with_judge(self):
        """Test: LLM judge should rate metrics quality highly."""
        # First get context
        context_agent = ContextDiscoveryAgent()
        context = context_agent.discover("AI Search")
        
        # Then generate framework
        framework_agent = SuccessFrameworkAgent()
        framework = framework_agent.generate(context)
        
        # Format metrics for judge
        primary_str = "\n".join([f"- {m.display_name}: {m.description}" 
                                  for m in framework.primary_metrics])
        secondary_str = "\n".join([f"- {m.display_name}: {m.description}" 
                                    for m in framework.secondary_metrics])
        
        # Use LLM as judge
        judge_prompt = JUDGE_PROMPT_METRICS_QUALITY.format(
            feature_name=context.name,
            primary_metrics=primary_str,
            secondary_metrics=secondary_str
        )
        
        judge_response = generate_text_sync(
            prompt=judge_prompt,
            trace_name="llm_judge_metrics_quality",
            trace_metadata={"feature": context.name}
        )
        
        # Extract score
        try:
            score = float(judge_response.strip().split('\n')[-1])
            assert score >= 7.0, f"Quality score {score} below threshold 7.0"
        except ValueError:
            pytest.skip(f"Could not parse judge score from: {judge_response}")


# Pytest markers for different test types
# Run fast tests: pytest evals/test_llm_components.py -m "not slow"
# Run all tests: pytest evals/test_llm_components.py
# Run with Langfuse tracing: pytest evals/test_llm_components.py -v

"""
Evaluation datasets for testing the Product Success Tracking Agent.

Why datasets?
- Consistent test cases across runs
- Easy to add new examples
- Can track performance over time
- Golden examples for regression testing

How to use:
1. Add test cases here
2. Write evaluators that use these cases
3. Run with pytest to see pass/fail
4. View results in Langfuse dashboard
"""

from typing import TypedDict


class ContextDiscoveryCase(TypedDict):
    """Test case for context discovery agent."""
    feature_name: str
    additional_context: str
    expected_keywords: list[str]  # Should appear in description/purpose
    expected_user_types: list[str]  # Should appear in target_users
    min_confidence: float


class SuccessFrameworkCase(TypedDict):
    """Test case for success framework generation."""
    feature_name: str
    feature_context: str
    expected_metric_types: list[str]  # e.g., ["conversion", "engagement", "retention"]
    min_primary_metrics: int
    min_secondary_metrics: int


class PipelineCase(TypedDict):
    """End-to-end pipeline test case."""
    feature_name: str
    additional_context: str
    expected_eligible_tables: list[str]  # Table names that should be found
    expected_grains: dict[str, str]  # table_name -> grain_type
    max_execution_time_ms: float


# --- Context Discovery Test Cases ---

CONTEXT_DISCOVERY_CASES: list[ContextDiscoveryCase] = [
    {
        "feature_name": "AI Search",
        "additional_context": "",
        "expected_keywords": ["search", "find", "query", "results"],
        "expected_user_types": ["all users", "power users", "researchers"],
        "min_confidence": 0.6
    },
    {
        "feature_name": "Collaborative Editing",
        "additional_context": "Real-time document collaboration",
        "expected_keywords": ["collaborate", "real-time", "edit", "share"],
        "expected_user_types": ["team", "enterprise", "collaborators"],
        "min_confidence": 0.6
    },
    {
        "feature_name": "Analytics Dashboard",
        "additional_context": "",
        "expected_keywords": ["analytics", "metrics", "dashboard", "insights"],
        "expected_user_types": ["analysts", "managers", "executives"],
        "min_confidence": 0.6
    }
]


# --- Success Framework Test Cases ---

SUCCESS_FRAMEWORK_CASES: list[SuccessFrameworkCase] = [
    {
        "feature_name": "AI Search",
        "feature_context": "Helps users find content faster using AI-powered search",
        "expected_metric_types": ["conversion", "engagement", "quality"],
        "min_primary_metrics": 3,
        "min_secondary_metrics": 2
    },
    {
        "feature_name": "Onboarding Flow",
        "feature_context": "Guides new users through product setup",
        "expected_metric_types": ["completion", "retention", "time"],
        "min_primary_metrics": 3,
        "min_secondary_metrics": 2
    }
]


# --- Database Explorer Test Cases ---

DATABASE_EXPLORER_CASES = [
    {
        "keywords": ["search", "event"],
        "expected_tables": ["search_events", "search_clicks"],
        "min_tables": 2
    },
    {
        "keywords": ["conversion", "user"],
        "expected_tables": ["conversions", "users"],
        "min_tables": 2
    }
]


# --- Grain Detection Test Cases ---

GRAIN_DETECTION_CASES = [
    {
        "table_name": "search_events",
        "expected_grain": "event",
        "min_confidence": 0.8
    },
    {
        "table_name": "users",
        "expected_grain": "user",
        "min_confidence": 0.8
    },
    {
        "table_name": "sessions",
        "expected_grain": "session",
        "min_confidence": 0.8
    }
]


# --- End-to-End Pipeline Test Cases ---

PIPELINE_CASES: list[PipelineCase] = [
    {
        "feature_name": "AI Search",
        "additional_context": "",
        "expected_eligible_tables": ["search_events", "search_clicks"],
        "expected_grains": {
            "search_events": "event",
            "search_clicks": "event"
        },
        "max_execution_time_ms": 15000  # 15 seconds
    }
]


# --- LLM-as-Judge Prompts ---

JUDGE_PROMPT_CONTEXT_QUALITY = """
Evaluate the quality of this feature context discovery (score 0-10):

Feature: {feature_name}
Description: {description}
Purpose: {purpose}
Target Users: {target_users}

Criteria:
1. Clarity: Is the description clear and specific?
2. Relevance: Does it match what you know about this feature?
3. Completeness: Are all key aspects covered?
4. Actionability: Can this guide metric selection?

Score (0-10):
"""

JUDGE_PROMPT_METRICS_QUALITY = """
Evaluate the quality of these success metrics (score 0-10):

Feature: {feature_name}
Primary Metrics:
{primary_metrics}

Secondary Metrics:
{secondary_metrics}

Criteria:
1. Specificity: Are metrics well-defined and measurable?
2. Relevance: Do they align with the feature purpose?
3. Actionability: Can teams act on these metrics?
4. Coverage: Do they cover key success dimensions?

Score (0-10):
"""


# --- Expected Outputs (Golden Examples) ---

GOLDEN_OUTPUTS = {
    "AI Search": {
        "primary_metrics": [
            "Search Conversion Rate",
            "Click-Through Rate (CTR)",
            "Zero Results Rate"
        ],
        "secondary_metrics": [
            "Search Usage Rate",
            "Average Results per Query",
            "Time to First Click"
        ]
    }
}

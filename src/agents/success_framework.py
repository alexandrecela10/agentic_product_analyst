"""
Success Framework Agent - Maps feature types to metric frameworks.

Given a FeatureContext, this agent recommends:
1. Primary success metrics (the key KPIs)
2. Secondary metrics (supporting indicators)
3. Tracking approach (what events to capture)
4. Analysis suggestions (how to analyze the data)

Why a separate agent?
- Context Discovery finds WHAT the feature is
- Success Framework determines HOW to measure it
- Separation allows different prompts optimized for each task
"""

from typing import Optional
from pydantic import BaseModel, Field
from src.agents.base_agent import BaseAgent
from src.agents.context_discovery import FeatureContext
from src.core.observability import get_logger

logger = get_logger(__name__)


class MetricDefinition(BaseModel):
    """
    Definition of a single metric to track.
    """
    name: str = Field(description="Metric name (e.g., 'search_conversion_rate')")
    display_name: str = Field(description="Human-readable name")
    description: str = Field(description="What this metric measures")
    formula: str = Field(description="How to calculate (e.g., 'conversions / searches')")
    target: Optional[str] = Field(default=None, description="Target value if known")
    frequency: str = Field(description="How often to measure (daily, weekly, etc.)")


class TrackingEvent(BaseModel):
    """
    An event that should be tracked for this feature.
    """
    event_name: str = Field(description="Event name (e.g., 'search_submitted')")
    trigger: str = Field(description="When this event fires")
    properties: list[str] = Field(description="Properties to capture with event")


class SuccessFramework(BaseModel):
    """
    Complete framework for measuring feature success.
    
    This is the output that guides the data analyst on
    what to track and how to analyze it.
    """
    feature_name: str = Field(description="Feature this framework is for")
    primary_metrics: list[MetricDefinition] = Field(description="Key success metrics (2-3)")
    secondary_metrics: list[MetricDefinition] = Field(description="Supporting metrics (3-5)")
    tracking_events: list[TrackingEvent] = Field(description="Events to implement")
    analysis_approach: str = Field(description="Recommended analysis methodology")
    data_requirements: list[str] = Field(description="Data needed for this analysis")
    caveats: list[str] = Field(default_factory=list, description="Limitations or considerations")


# System prompt for Success Framework Agent
SUCCESS_FRAMEWORK_SYSTEM_PROMPT = """You are a Product Analytics Expert specializing in SaaS metrics and feature success measurement.

Given context about a feature, create a comprehensive success measurement framework.

GUIDELINES:
1. Primary metrics should directly measure the feature's core purpose (2-3 metrics max)
2. Secondary metrics provide supporting context (3-5 metrics)
3. Events should be specific and actionable for engineering to implement
4. Analysis approach should be practical for a data analyst to execute

METRIC NAMING CONVENTIONS:
- Use snake_case for metric names (e.g., search_conversion_rate)
- Use clear display names (e.g., "Search Conversion Rate")
- Include units in formulas where applicable

COMMON SAAS METRIC PATTERNS:
- Usage features: adoption_rate, active_users, session_depth, feature_stickiness
- Conversion features: conversion_rate, funnel_completion, time_to_convert
- Engagement features: dau_mau_ratio, return_rate, session_frequency
- Search features: zero_result_rate, click_through_rate, search_to_conversion

OUTPUT FORMAT:
Return a JSON object with this structure:
```json
{
    "feature_name": "Feature Name",
    "primary_metrics": [
        {
            "name": "metric_name",
            "display_name": "Metric Display Name",
            "description": "What it measures",
            "formula": "numerator / denominator",
            "target": "3-5%" or null,
            "frequency": "daily"
        }
    ],
    "secondary_metrics": [...],
    "tracking_events": [
        {
            "event_name": "event_name",
            "trigger": "When user does X",
            "properties": ["property1", "property2"]
        }
    ],
    "analysis_approach": "Description of how to analyze...",
    "data_requirements": ["requirement1", "requirement2"],
    "caveats": ["caveat1", "caveat2"]
}
```"""


class SuccessFrameworkAgent(BaseAgent[SuccessFramework]):
    """
    Generates a success measurement framework for a feature.
    
    Takes FeatureContext as input and produces a complete
    framework for tracking and analyzing the feature.
    """
    
    def __init__(self):
        super().__init__(
            name="success_framework",
            system_prompt=SUCCESS_FRAMEWORK_SYSTEM_PROMPT
        )
    
    def _build_prompt(self, context: FeatureContext) -> str:
        """
        Build prompt from FeatureContext.
        """
        # Format context for the prompt
        prompt = f"""Create a success measurement framework for the following feature:

FEATURE CONTEXT:
- Name: {context.name}
- Description: {context.description}
- Purpose: {context.purpose}
- Target Users: {', '.join(context.target_users)}
- Success Criteria from PRD: {', '.join(context.success_criteria) if context.success_criteria else 'Not specified'}
- Recommended Metrics (from context): {', '.join(context.recommended_metrics) if context.recommended_metrics else 'Not specified'}
- Industry Benchmarks: {context.industry_benchmarks if context.industry_benchmarks else 'Not available'}

Based on this context, create a comprehensive success measurement framework.
Focus on metrics that directly measure whether the feature achieves its stated purpose.

Return your framework as a JSON object."""
        
        return prompt
    
    def _parse_response(self, response: str) -> SuccessFramework:
        """
        Parse LLM response into SuccessFramework.
        """
        data = self._extract_json(response)
        return SuccessFramework(**data)
    
    def generate(self, context: FeatureContext) -> SuccessFramework:
        """
        Generate a success framework for a feature.
        
        Args:
            context: FeatureContext from context discovery
            
        Returns:
            SuccessFramework with metrics and tracking plan
        """
        return self.run(context=context)


def generate_success_framework(context: FeatureContext) -> SuccessFramework:
    """
    Convenience function to generate a success framework.
    
    Args:
        context: FeatureContext from context discovery
        
    Returns:
        SuccessFramework with metrics and tracking plan
    """
    agent = SuccessFrameworkAgent()
    return agent.generate(context)

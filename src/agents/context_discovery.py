"""
Context Discovery Agent - RAG-powered feature context extraction.

This agent answers: "What is this feature and how should we measure its success?"

It uses RAG to:
1. Search company docs (PRDs, specs) for feature information
2. Optionally search the web for industry benchmarks
3. Synthesize findings into a structured FeatureContext

Why RAG here?
- Company docs contain the ground truth about features
- Web search provides industry context and benchmarks
- LLM synthesizes multiple sources into coherent understanding
"""

from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field
from src.agents.base_agent import BaseAgent
from src.rag.retriever import HybridRetriever, get_retriever, RetrievalResult
from src.core.observability import get_logger

logger = get_logger(__name__)


class FeatureContext(BaseModel):
    """
    Structured output from context discovery.
    
    This captures everything a data analyst needs to know
    before starting to track a feature's success.
    """
    name: str = Field(description="Feature name as identified")
    description: str = Field(description="Brief description of what the feature does")
    purpose: str = Field(description="The problem this feature solves")
    target_users: list[str] = Field(description="Who uses this feature")
    success_criteria: list[str] = Field(description="How success was defined (from PRD or inferred)")
    recommended_metrics: list[str] = Field(description="KPIs to track for this feature")
    related_features: list[str] = Field(default_factory=list, description="Related or dependent features")
    industry_benchmarks: dict = Field(default_factory=dict, description="Industry benchmark values if found")
    confidence: float = Field(description="Confidence score 0-1 based on source quality")
    sources: list[str] = Field(description="Documents used to generate this context")


# System prompt for the Context Discovery Agent
CONTEXT_DISCOVERY_SYSTEM_PROMPT = """You are a Product Analytics Expert helping data analysts understand features before tracking them.

Your task is to analyze retrieved documents about a feature and extract structured context.

IMPORTANT GUIDELINES:
1. Only use information from the provided documents - do not make up facts
2. If information is missing, say "Not specified in documents" rather than guessing
3. For success_criteria, extract explicit metrics from PRDs if available
4. For recommended_metrics, suggest appropriate KPIs based on feature type
5. Set confidence based on how much relevant information was found (0.0-1.0)

OUTPUT FORMAT:
Return a JSON object with this exact structure:
```json
{
    "name": "Feature Name",
    "description": "What the feature does",
    "purpose": "The problem it solves",
    "target_users": ["User type 1", "User type 2"],
    "success_criteria": ["Metric 1 target", "Metric 2 target"],
    "recommended_metrics": ["metric_1", "metric_2", "metric_3"],
    "related_features": ["Related feature 1"],
    "industry_benchmarks": {"metric_name": "benchmark_value"},
    "confidence": 0.85,
    "sources": ["document_name_1.md", "document_name_2.md"]
}
```

Focus on actionable insights that help a data analyst set up tracking."""


class ContextDiscoveryAgent(BaseAgent[FeatureContext]):
    """
    Discovers context about a feature using RAG.
    
    Workflow:
    1. Retrieve relevant chunks from knowledge base
    2. Format chunks as context for LLM
    3. LLM extracts structured FeatureContext
    """
    
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        top_k: int = 5
    ):
        """
        Initialize the agent.
        
        Args:
            retriever: Hybrid retriever for RAG (default: singleton)
            top_k: Number of chunks to retrieve
        """
        super().__init__(
            name="context_discovery",
            system_prompt=CONTEXT_DISCOVERY_SYSTEM_PROMPT
        )
        self.retriever = retriever or get_retriever()
        self.top_k = top_k
    
    def _build_prompt(self, feature_name: str, additional_context: str = "") -> str:
        """
        Build prompt with retrieved context.
        
        Args:
            feature_name: Name of the feature to discover
            additional_context: Optional user-provided context
        """
        # Retrieve relevant chunks
        query = f"feature: {feature_name} purpose success metrics tracking"
        results = self.retriever.retrieve(query, top_k=self.top_k)
        
        # Store results for later reference
        self._last_retrieval = results
        
        # Format retrieved context
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.metadata.get('document_name', 'Unknown')
            section = result.metadata.get('section_title', '')
            
            context_parts.append(f"""
--- Document {i}: {source} {f'(Section: {section})' if section else ''} ---
{result.content}
""")
        
        retrieved_context = "\n".join(context_parts)
        
        # Build final prompt
        prompt = f"""Analyze the following documents to extract context about the feature: "{feature_name}"

{f'Additional context from user: {additional_context}' if additional_context else ''}

RETRIEVED DOCUMENTS:
{retrieved_context}

Based on these documents, extract structured information about "{feature_name}".
If the documents don't contain enough information, set confidence lower and note what's missing.

Return your analysis as a JSON object."""
        
        return prompt
    
    def _parse_response(self, response: str) -> FeatureContext:
        """
        Parse LLM response into FeatureContext.
        """
        data = self._extract_json(response)
        return FeatureContext(**data)
    
    def discover(
        self,
        feature_name: str,
        additional_context: str = ""
    ) -> FeatureContext:
        """
        Discover context for a feature.
        
        Args:
            feature_name: Name of the feature (e.g., "AI Search")
            additional_context: Optional additional context from user
            
        Returns:
            FeatureContext with extracted information
        """
        return self.run(
            feature_name=feature_name,
            additional_context=additional_context
        )
    
    def get_last_retrieval(self) -> list[RetrievalResult]:
        """
        Get the chunks retrieved in the last discovery.
        
        Useful for debugging and understanding what context was used.
        """
        return getattr(self, '_last_retrieval', [])


def discover_feature_context(
    feature_name: str,
    additional_context: str = ""
) -> FeatureContext:
    """
    Convenience function to discover feature context.
    
    Args:
        feature_name: Name of the feature
        additional_context: Optional additional context
        
    Returns:
        FeatureContext with extracted information
    """
    agent = ContextDiscoveryAgent()
    return agent.discover(feature_name, additional_context)

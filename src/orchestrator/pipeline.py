"""
Pipeline Orchestrator - Coordinates all components.

This is the main entry point that chains together:
1. Context Discovery (RAG + LLM)
2. Success Framework (LLM)
3. Database Explorer (Deterministic)
4. Grain Detector (Deterministic)

Why an orchestrator?
- Single entry point for the entire workflow
- Manages state between components
- Handles errors gracefully
- Provides observability across the pipeline
"""

from typing import Optional
from datetime import datetime
import time
from src.agents.context_discovery import ContextDiscoveryAgent, FeatureContext
from src.agents.success_framework import SuccessFrameworkAgent, SuccessFramework
from src.deterministic.database_explorer import DatabaseExplorer
from src.deterministic.grain_detector import GrainDetector, detect_grain_from_model
from src.rag.indexer import DocumentIndexer
from src.core.observability import get_logger, init_observability
from src.core.models import (
    PipelineInput,
    PipelineOutput,
    FeatureContextModel,
    SuccessFrameworkModel,
    MetricDefinitionModel,
    TrackingEventModel,
    ScoredTableModel,
    GrainResultModel,
    GrainType,
    ErrorDetail,
    PipelineError
)

logger = get_logger(__name__)


class ProductSuccessPipeline:
    """
    Main orchestrator for the Product Success Tracking Agent.
    
    Workflow:
    1. Index knowledge base (if needed)
    2. Discover feature context via RAG
    3. Generate success framework
    4. Find eligible tables
    5. Detect grain for each table
    """
    
    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        data_directory: str = "./data/mock_warehouse"
    ):
        """
        Initialize the pipeline.
        
        Args:
            knowledge_base_path: Path to documents for RAG
            data_directory: Path to CSV files (mock warehouse)
        """
        self.knowledge_base_path = knowledge_base_path
        self.data_directory = data_directory
        
        # Initialize components
        self.indexer = DocumentIndexer(knowledge_base_path=knowledge_base_path)
        self.context_agent = ContextDiscoveryAgent()
        self.framework_agent = SuccessFrameworkAgent()
        self.db_explorer = DatabaseExplorer(data_directory=data_directory)
        self.grain_detector = GrainDetector()
        
        # Track LLM calls
        self._llm_calls = 0
    
    def index_knowledge_base(self) -> dict:
        """
        Index the knowledge base for RAG.
        
        Call this once before running the pipeline,
        or when documents change.
        """
        logger.info("indexing_knowledge_base", path=self.knowledge_base_path)
        return self.indexer.index_all()
    
    def run(self, input_data: PipelineInput) -> PipelineOutput:
        """
        Run the complete pipeline for a feature.
        
        Args:
            input_data: PipelineInput with feature_name and options
            
        Returns:
            PipelineOutput (Pydantic) with all analysis
        """
        start_time = time.time()
        errors = []
        self._llm_calls = 0
        
        feature_name = input_data.feature_name
        additional_context = input_data.additional_context
        skip_indexing = input_data.skip_indexing
        
        logger.info("pipeline_started", feature=feature_name)
        
        # Step 0: Index knowledge base (if needed)
        if not skip_indexing:
            try:
                self.index_knowledge_base()
            except Exception as e:
                errors.append(f"Indexing error: {e}")
                logger.error("indexing_failed", error=str(e))
        
        # Step 1: Context Discovery (RAG + LLM)
        logger.info("step_1_context_discovery")
        try:
            feature_context_raw = self.context_agent.discover(
                feature_name=feature_name,
                additional_context=additional_context
            )
            self._llm_calls += 1
            
            # Convert to Pydantic model
            feature_context = FeatureContextModel(
                name=feature_context_raw.name,
                description=feature_context_raw.description,
                purpose=feature_context_raw.purpose,
                target_users=feature_context_raw.target_users,
                success_criteria=feature_context_raw.success_criteria,
                recommended_metrics=feature_context_raw.recommended_metrics,
                related_features=feature_context_raw.related_features,
                industry_benchmarks=feature_context_raw.industry_benchmarks,
                confidence=feature_context_raw.confidence,
                sources=feature_context_raw.sources
            )
        except PipelineError as e:
            errors.append(f"Context discovery error: {e.error.message}")
            logger.error("context_discovery_failed", error=e.error.message, code=e.error.code)
            feature_context = self._create_fallback_context(feature_name)
        except Exception as e:
            errors.append(f"Context discovery error: {e}")
            logger.error("context_discovery_failed", error=str(e))
            feature_context = self._create_fallback_context(feature_name)
        
        # Step 2: Success Framework (LLM)
        logger.info("step_2_success_framework")
        try:
            # Convert back to agent's expected input format
            context_for_agent = FeatureContext(
                name=feature_context.name,
                description=feature_context.description,
                purpose=feature_context.purpose,
                target_users=feature_context.target_users,
                success_criteria=feature_context.success_criteria,
                recommended_metrics=feature_context.recommended_metrics,
                related_features=feature_context.related_features,
                industry_benchmarks=feature_context.industry_benchmarks,
                confidence=feature_context.confidence,
                sources=feature_context.sources
            )
            
            framework_raw = self.framework_agent.generate(context_for_agent)
            self._llm_calls += 1
            
            # Convert to Pydantic model
            success_framework = SuccessFrameworkModel(
                feature_name=framework_raw.feature_name,
                primary_metrics=[
                    MetricDefinitionModel(
                        name=m.name,
                        display_name=m.display_name,
                        description=m.description,
                        formula=m.formula,
                        target=m.target,
                        frequency=m.frequency
                    ) for m in framework_raw.primary_metrics
                ],
                secondary_metrics=[
                    MetricDefinitionModel(
                        name=m.name,
                        display_name=m.display_name,
                        description=m.description,
                        formula=m.formula,
                        target=m.target,
                        frequency=m.frequency
                    ) for m in framework_raw.secondary_metrics
                ],
                tracking_events=[
                    TrackingEventModel(
                        event_name=e.event_name,
                        trigger=e.trigger,
                        properties=e.properties
                    ) for e in framework_raw.tracking_events
                ],
                analysis_approach=framework_raw.analysis_approach,
                data_requirements=framework_raw.data_requirements,
                caveats=framework_raw.caveats
            )
        except PipelineError as e:
            errors.append(f"Success framework error: {e.error.message}")
            logger.error("success_framework_failed", error=e.error.message)
            success_framework = self._create_fallback_framework(feature_name)
        except Exception as e:
            errors.append(f"Success framework error: {e}")
            logger.error("success_framework_failed", error=str(e))
            success_framework = self._create_fallback_framework(feature_name)
        
        # Step 3: Database Exploration (Deterministic - no LLM)
        logger.info("step_3_database_exploration")
        eligible_tables: list[ScoredTableModel] = []
        try:
            # Extract keywords from context for table matching
            keywords = [feature_name.lower()]
            keywords.extend([m.name for m in success_framework.primary_metrics])
            keywords.extend([e.event_name for e in success_framework.tracking_events])
            
            # Returns list of ScoredTableModel (already Pydantic)
            eligible_tables = self.db_explorer.find_eligible_tables(keywords)
        except PipelineError as e:
            errors.append(f"Database exploration error: {e.error.message}")
            logger.error("db_exploration_failed", error=e.error.message)
        except Exception as e:
            errors.append(f"Database exploration error: {e}")
            logger.error("db_exploration_failed", error=str(e))
        
        # Step 4: Grain Detection (Deterministic - no LLM)
        logger.info("step_4_grain_detection")
        grain_results: dict[str, GrainResultModel] = {}
        for scored_table in eligible_tables:
            try:
                # Use the Pydantic model directly
                grain = detect_grain_from_model(scored_table.table)
                grain_results[scored_table.table.name] = grain
            except PipelineError as e:
                errors.append(f"Grain detection error for {scored_table.table.name}: {e.error.message}")
                logger.error("grain_detection_failed", table=scored_table.table.name, error=e.error.message)
            except Exception as e:
                errors.append(f"Grain detection error for {scored_table.table.name}: {e}")
                logger.error("grain_detection_failed", table=scored_table.table.name, error=str(e))
        
        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Build result using Pydantic model
        result = PipelineOutput(
            feature_context=feature_context,
            success_framework=success_framework,
            eligible_tables=eligible_tables,
            grain_results=grain_results,
            llm_calls=self._llm_calls,
            errors=errors,
            execution_time_ms=execution_time_ms,
            timestamp=datetime.utcnow()
        )
        
        logger.info("pipeline_completed",
                   feature=feature_name,
                   llm_calls=self._llm_calls,
                   tables_found=len(eligible_tables),
                   errors=len(errors),
                   execution_time_ms=execution_time_ms)
        
        return result
    
    def _create_fallback_context(self, feature_name: str) -> FeatureContextModel:
        """Create minimal context when discovery fails."""
        return FeatureContextModel(
            name=feature_name,
            description="Could not discover context",
            purpose="Unknown",
            target_users=[],
            success_criteria=[],
            recommended_metrics=[],
            confidence=0.0,
            sources=[]
        )
    
    def _create_fallback_framework(self, feature_name: str) -> SuccessFrameworkModel:
        """Create minimal framework when generation fails."""
        return SuccessFrameworkModel(
            feature_name=feature_name,
            primary_metrics=[],
            secondary_metrics=[],
            tracking_events=[],
            analysis_approach="Could not generate framework",
            data_requirements=[]
        )


def run_pipeline(
    feature_name: str,
    additional_context: str = ""
) -> PipelineOutput:
    """
    Convenience function to run the pipeline.
    
    Args:
        feature_name: Name of the feature to analyze
        additional_context: Optional additional context
        
    Returns:
        PipelineOutput (Pydantic) with structured results
    """
    init_observability()
    pipeline = ProductSuccessPipeline()
    
    # Create structured input
    input_data = PipelineInput(
        feature_name=feature_name,
        additional_context=additional_context
    )
    
    return pipeline.run(input_data)

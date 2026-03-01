"""
Base agent class for LLM-backed agents.

Provides common functionality for all agents:
- LLM interaction with Gemini
- Observability integration (Langfuse tracing)
- Structured output parsing

Why a base class?
- DRY: Don't repeat LLM setup in every agent
- Consistent observability across all agents
- Easy to swap LLM providers
"""

from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Generic
from pydantic import BaseModel
import json
import re
from src.core.llm_client import generate_text_sync, get_llm_model
from src.core.observability import get_logger
from src.core.config import settings

logger = get_logger(__name__)

# Generic type for agent output
T = TypeVar('T', bound=BaseModel)


class BaseAgent(ABC, Generic[T]):
    """
    Abstract base class for LLM-backed agents.
    
    Each agent:
    1. Has a specific task (context discovery, success framework, etc.)
    2. Uses a system prompt to guide behavior
    3. Returns structured output (Pydantic model)
    """
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        model_name: Optional[str] = None
    ):
        """
        Initialize the agent.
        
        Args:
            name: Agent name for logging/tracing
            system_prompt: Instructions for the LLM
            model_name: Override default model from settings
        """
        self.name = name
        self.system_prompt = system_prompt
        self.model_name = model_name or settings.gemini_model
    
    @abstractmethod
    def _build_prompt(self, **kwargs) -> str:
        """
        Build the user prompt for the LLM.
        
        Each agent implements this to format its specific input.
        """
        pass
    
    @abstractmethod
    def _parse_response(self, response: str) -> T:
        """
        Parse the LLM response into structured output.
        
        Each agent implements this to extract its specific output type.
        """
        pass
    
    def run(self, **kwargs) -> T:
        """
        Execute the agent.
        
        Workflow:
        1. Build prompt from inputs
        2. Call LLM with system prompt
        3. Parse response into structured output
        4. Log to Langfuse for observability
        
        Returns:
            Structured output (Pydantic model)
        """
        # Build the prompt
        prompt = self._build_prompt(**kwargs)
        
        logger.info(f"{self.name}_started", prompt_length=len(prompt))
        
        # Call LLM with built-in Langfuse tracing
        # The generate_text_sync function now handles all observability
        response = generate_text_sync(
            prompt=prompt,
            system_instruction=self.system_prompt,
            model_name=self.model_name,
            trace_name=self.name,  # This appears in Langfuse dashboard
            trace_metadata={"agent": self.name, **kwargs}  # Include inputs for debugging
        )
        
        logger.info(f"{self.name}_completed", response_length=len(response))
        
        # Parse response
        result = self._parse_response(response)
        
        return result
    
    def _extract_json(self, text: str) -> dict:
        """
        Extract JSON from LLM response.
        
        LLMs often wrap JSON in markdown code blocks.
        This handles both raw JSON and ```json ... ``` blocks.
        """
        # Try to find JSON in code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try raw JSON
            json_str = text.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("json_parse_error", error=str(e), text=text[:500])
            raise ValueError(f"Failed to parse JSON from LLM response: {e}")

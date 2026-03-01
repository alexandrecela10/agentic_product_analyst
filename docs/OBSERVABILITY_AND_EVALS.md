# Observability & Evaluations Guide

This guide explains how to observe and evaluate your AI system to master AI engineering.

## Table of Contents
1. [Observability Setup](#observability-setup)
2. [What Gets Traced](#what-gets-traced)
3. [Reading Langfuse Traces](#reading-langfuse-traces)
4. [Evaluation Framework](#evaluation-framework)
5. [Running Evaluations](#running-evaluations)

---

## Observability Setup

### What is Observability?

**Observability** means being able to see what your AI system is doing:
- What prompts are being sent to the LLM
- What responses come back
- How long each call takes
- What context is retrieved from RAG
- Where errors occur

### Why Langfuse?

Langfuse is purpose-built for LLM observability:
- **Traces**: Group related operations (e.g., one pipeline run)
- **Generations**: Individual LLM calls with input/output
- **Spans**: Non-LLM operations (DB queries, API calls)
- **Scores**: Evaluation results attached to traces

### Current Setup

Every LLM call in this project is automatically traced:

```python
# In src/core/llm_client.py
response = generate_text_sync(
    prompt="Your prompt here",
    trace_name="context_discovery",  # Shows up in Langfuse
    trace_metadata={"feature": "AI Search"}  # Additional context
)
```

**What happens:**
1. Prompt is sent to Gemini
2. Response is received
3. Both are logged to Langfuse with:
   - Input (prompt + system instruction)
   - Output (response text)
   - Model used (gemini-2.0-flash)
   - Latency in milliseconds
   - Metadata you provide

---

## What Gets Traced

### 1. LLM Calls (Generations)

**Location**: `src/core/llm_client.py`

Every call to `generate_text_sync()` creates a Langfuse generation:

```python
# Example: Context Discovery Agent
response = generate_text_sync(
    prompt=f"Analyze this feature: {feature_name}",
    system_instruction="You are a product analyst...",
    trace_name="context_discovery",
    trace_metadata={
        "feature": feature_name,
        "agent": "ContextDiscoveryAgent"
    }
)
```

**In Langfuse you'll see:**
- Trace name: `context_discovery`
- Input: Full prompt + system instruction
- Output: LLM response
- Metadata: Feature name, agent name
- Latency: Time taken

### 2. RAG Retrieval (Spans)

**Location**: `src/rag/vector_store.py`, `src/rag/retriever.py`

When you add tracing to retrieval:

```python
# Vector search
results = store.search("search conversion metrics", top_k=5)
# This should log:
# - Query text
# - Retrieved chunks
# - Similarity scores
```

### 3. Pipeline Execution

**Location**: `src/orchestrator/pipeline.py`

The full pipeline creates a hierarchical trace:

```
Pipeline Run (trace)
├── Context Discovery (generation)
├── Success Framework (generation)
├── Database Explorer (span)
└── Grain Detection (span)
```

---

## Reading Langfuse Traces

### Dashboard Overview

1. **Go to**: https://cloud.langfuse.com
2. **Login** with your credentials
3. **Select your project**

### Traces Tab

Shows all pipeline runs:
- **Name**: What operation ran (e.g., "context_discovery")
- **Timestamp**: When it ran
- **Duration**: Total time
- **Status**: Success/Error
- **Metadata**: Custom fields you logged

### Generations Tab

Shows all LLM calls:
- **Model**: Which model was used
- **Input**: The prompt sent
- **Output**: The response received
- **Tokens**: Input/output token counts
- **Cost**: Estimated API cost
- **Latency**: Response time

### Drill Down

Click any trace to see:
1. **Timeline**: Visual representation of operations
2. **Input/Output**: Full prompts and responses
3. **Metadata**: Context you logged
4. **Scores**: Evaluation results (if any)

---

## Evaluation Framework

### What is Evaluation?

**Evaluation** means measuring how well your AI system performs:
- **Accuracy**: Does it give correct answers?
- **Relevance**: Is the retrieved context relevant?
- **Consistency**: Same input → same output?
- **Latency**: Is it fast enough?

### Evaluation Types

#### 1. Unit Evals (Component-Level)

Test individual components:

```python
# Test: Does context discovery find the right info?
def test_context_discovery():
    agent = ContextDiscoveryAgent()
    result = agent.discover("AI Search")
    
    assert result.confidence > 0.7
    assert "search" in result.description.lower()
    assert len(result.target_users) > 0
```

#### 2. Integration Evals (Pipeline-Level)

Test the full pipeline:

```python
# Test: Does the pipeline produce valid output?
def test_full_pipeline():
    result = run_pipeline("AI Search")
    
    assert len(result.success_framework.primary_metrics) >= 3
    assert len(result.eligible_tables) > 0
    assert result.execution_time_ms < 10000  # Under 10s
```

#### 3. LLM-as-Judge Evals

Use an LLM to evaluate outputs:

```python
# Test: Is the success framework high quality?
def eval_success_framework_quality(output):
    judge_prompt = f'''
    Evaluate this success framework for quality (0-10):
    
    {output.success_framework}
    
    Criteria:
    - Are metrics specific and measurable?
    - Do they align with the feature purpose?
    - Are they actionable?
    '''
    
    score = llm_judge(judge_prompt)
    return score >= 7  # Pass if score is 7+
```

#### 4. Golden Dataset Evals

Compare against known good outputs:

```python
# Test: Does it match expected output?
golden_examples = [
    {
        "input": "AI Search",
        "expected_metrics": ["Search Conversion Rate", "CTR", "Zero Results Rate"]
    }
]

def test_against_golden():
    for example in golden_examples:
        result = run_pipeline(example["input"])
        actual_metrics = [m.display_name for m in result.success_framework.primary_metrics]
        
        # Check overlap
        overlap = set(actual_metrics) & set(example["expected_metrics"])
        assert len(overlap) >= 2  # At least 2 matches
```

---

## Running Evaluations

### Setup Evaluation Dataset

Create test cases in `evals/datasets/`:

```python
# evals/datasets/context_discovery.py
CONTEXT_DISCOVERY_CASES = [
    {
        "feature_name": "AI Search",
        "expected_purpose": "help users find content faster",
        "expected_users": ["all users", "power users"],
        "min_confidence": 0.7
    },
    {
        "feature_name": "Collaborative Editing",
        "expected_purpose": "enable real-time collaboration",
        "expected_users": ["team users", "enterprise"],
        "min_confidence": 0.7
    }
]
```

### Write Evaluation Functions

Create evaluators in `evals/`:

```python
# evals/test_context_discovery.py
from src.agents.context_discovery import ContextDiscoveryAgent
from evals.datasets.context_discovery import CONTEXT_DISCOVERY_CASES
from langfuse import Langfuse

langfuse = Langfuse()

def test_context_discovery_accuracy():
    """Test if context discovery finds correct information."""
    agent = ContextDiscoveryAgent()
    
    for case in CONTEXT_DISCOVERY_CASES:
        # Run the agent
        result = agent.discover(case["feature_name"])
        
        # Evaluate
        passed = (
            result.confidence >= case["min_confidence"] and
            any(user.lower() in result.target_users.lower() 
                for user in case["expected_users"])
        )
        
        # Log to Langfuse
        langfuse.score(
            name="context_discovery_accuracy",
            value=1.0 if passed else 0.0,
            trace_id=result.trace_id  # Link to the trace
        )
        
        assert passed, f"Failed for {case['feature_name']}"
```

### Run Evals

```bash
# Run all evaluations
uv run pytest evals/

# Run specific eval
uv run pytest evals/test_context_discovery.py

# Run with verbose output
uv run pytest evals/ -v
```

### View Results in Langfuse

1. Go to **Scores** tab
2. Filter by score name (e.g., "context_discovery_accuracy")
3. See pass/fail rates over time
4. Click individual scores to see the trace

---

## Best Practices

### 1. Trace Everything Important

```python
# Good: Detailed tracing
response = generate_text_sync(
    prompt=prompt,
    trace_name="success_framework_generation",
    trace_metadata={
        "feature": feature_name,
        "context_confidence": context.confidence,
        "num_metrics_requested": 5
    }
)

# Bad: No context
response = generate_text_sync(prompt=prompt)
```

### 2. Use Meaningful Names

```python
# Good: Descriptive names
trace_name="context_discovery_ai_search"

# Bad: Generic names
trace_name="llm_call_1"
```

### 3. Log Intermediate Steps

```python
# Log retrieval results
logger.info("rag_retrieval", 
            query=query,
            num_results=len(results),
            top_score=results[0].score if results else 0)
```

### 4. Create Regression Tests

After fixing a bug, add an eval:

```python
def test_handles_empty_database():
    """Regression test: Should handle empty database gracefully."""
    result = run_pipeline("New Feature", data_directory="./empty")
    assert result.errors == []
    assert len(result.eligible_tables) == 0
```

### 5. Monitor Over Time

Track metrics across runs:
- Average latency
- Success rate
- Cost per run
- Token usage

---

## Next Steps

1. **Explore Langfuse**: Run a pipeline and explore the traces
2. **Write Your First Eval**: Start with a simple unit test
3. **Add Golden Examples**: Create expected outputs for key features
4. **Set Up CI**: Run evals on every commit
5. **Monitor Production**: Use Langfuse in production to catch issues

## Resources

- [Langfuse Docs](https://langfuse.com/docs)
- [Langfuse Python SDK](https://langfuse.com/docs/sdk/python)
- [LLM Evaluation Guide](https://www.langfuse.com/guides/cookbook/evaluation)

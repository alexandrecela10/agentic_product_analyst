# Getting Started - Product Success Tracking Agent

## What You've Built

A production-ready **multi-agent system** that answers: *"How do I track and quantify success for X product/feature?"*

**Architecture:**
- **RAG-powered Context Discovery** (LLM): Searches company docs to understand features
- **Success Framework Generation** (LLM): Recommends KPIs and tracking events
- **Database Explorer** (Deterministic): Finds relevant tables via schema introspection
- **Grain Detector** (Deterministic): Detects data granularity (event/session/user/firm/time)
- **Full Observability**: Every LLM call traced to Langfuse
- **Evaluation Framework**: Automated tests for quality assurance

## Quick Start

### 1. Test the System

```bash
# Run deterministic tests (fast, no API calls)
uv run pytest evals/test_deterministic.py -v

# Run a simple LLM call with tracing
uv run python -c "
from src.core.observability import init_observability
from src.core.llm_client import generate_text_sync

init_observability()

response = generate_text_sync(
    prompt='What are the top 3 metrics for a search feature?',
    trace_name='test_search_metrics',
    trace_metadata={'test': True}
)

print(response)
print('\n✅ Check Langfuse: https://cloud.langfuse.com')
"
```

### 2. Run the Full Pipeline

```bash
# CLI mode (quick test)
uv run python main.py cli "AI Search"

# This will:
# 1. Discover context from docs (RAG + LLM)
# 2. Generate success framework (LLM)
# 3. Find eligible tables (deterministic)
# 4. Detect grain for each table (deterministic)
# 5. Output results to console + JSON file
```

### 3. Start the API Server

```bash
# Start FastAPI server
uv run python main.py api

# API docs at: http://localhost:8000/docs
```

**Key Endpoints:**
- `POST /pipeline/run` - Run full pipeline
- `POST /rag/index` - Index knowledge base
- `POST /database/explore` - Explore tables
- `POST /grain/detect` - Detect table grain

### 4. View Traces in Langfuse

1. Go to https://cloud.langfuse.com
2. Select your project
3. Click **Traces** tab
4. You'll see every LLM call with:
   - Input prompt
   - Output response
   - Latency
   - Model used
   - Metadata

## Understanding the System

### Component Breakdown

| Component | Type | Purpose | Why This Approach? |
|-----------|------|---------|-------------------|
| **Context Discovery** | RAG + LLM | Extract feature info from docs | Requires understanding unstructured text |
| **Success Framework** | LLM | Map feature type → metrics | Needs domain knowledge and reasoning |
| **Database Explorer** | Deterministic | Find relevant tables | Pure pattern matching, no LLM needed |
| **Grain Detector** | Deterministic | Detect data granularity | Finite patterns + cardinality math |

**Key Principle:** *"LLM only when needed, deterministic otherwise"*

### LLM Calls Per Request

Typical pipeline run makes **2-3 LLM calls**:
1. Context Discovery (1 call)
2. Success Framework Generation (1 call)
3. Optional: Web search if docs insufficient (1 call)

Everything else is deterministic (fast, free, predictable).

### Observability Flow

```
User Request
    ↓
Pipeline Starts → Langfuse Trace Created
    ↓
Context Discovery Agent
    ↓
    LLM Call → Logged to Langfuse
        - Input: prompt + system instruction
        - Output: structured context
        - Latency: ~2-3 seconds
    ↓
Success Framework Agent
    ↓
    LLM Call → Logged to Langfuse
        - Input: context + prompt
        - Output: metrics framework
        - Latency: ~2-3 seconds
    ↓
Database Explorer (deterministic)
    ↓
Grain Detector (deterministic)
    ↓
Pipeline Complete → Results + Trace ID
```

## Evaluation Framework

### Test Structure

```
evals/
├── datasets.py              # Test cases and golden examples
├── test_deterministic.py    # Fast tests (no API calls)
├── test_llm_components.py   # LLM-powered tests (requires API)
└── test_pipeline.py         # End-to-end integration tests
```

### Running Evaluations

```bash
# Run all tests
uv run pytest evals/ -v

# Run only fast tests (no LLM calls)
uv run pytest evals/test_deterministic.py -v

# Run specific test
uv run pytest evals/test_deterministic.py::TestGrainDetector::test_grain_detection -v

# Stop on first failure
uv run pytest evals/ -x

# Show print statements
uv run pytest evals/ -s
```

### What Gets Tested

**Deterministic Components (10 tests):**
- ✅ Database explorer finds all tables
- ✅ Column metadata extraction
- ✅ Eligible table scoring
- ✅ Grain detection accuracy
- ✅ Data quality checks

**LLM Components (when you run them):**
- Context discovery keyword matching
- Confidence score validation
- Success framework metric count
- Structured output parsing
- LLM-as-judge quality evaluation

**End-to-End Pipeline:**
- Full pipeline execution
- Performance benchmarks
- Golden example matching
- Error handling

## Next Steps

### 1. Explore Langfuse Dashboard

**What to look for:**
- **Traces**: See the full pipeline execution
- **Generations**: Individual LLM calls with prompts/responses
- **Latency**: How long each call takes
- **Costs**: Estimated API costs

**Try this:**
1. Run: `uv run python main.py cli "AI Search"`
2. Go to Langfuse dashboard
3. Find the trace named "context_discovery"
4. Click to see the full prompt and response
5. Check the metadata to see feature name

### 2. Add Your Own Test Cases

Edit `evals/datasets.py`:

```python
CONTEXT_DISCOVERY_CASES.append({
    "feature_name": "Your Feature Name",
    "additional_context": "",
    "expected_keywords": ["keyword1", "keyword2"],
    "expected_user_types": ["user_type1"],
    "min_confidence": 0.6
})
```

Then run: `uv run pytest evals/test_llm_components.py -v`

### 3. Add Your Own Documents

Add markdown files to `knowledge_base/company_docs/`:

```bash
# Create a new PRD
cat > knowledge_base/company_docs/my_feature.md << 'EOF'
# My Feature

## Overview
Description of your feature...

## Success Metrics
- Metric 1
- Metric 2
EOF

# Index the knowledge base
uv run python -c "
from src.rag.indexer import index_knowledge_base
stats = index_knowledge_base('./knowledge_base')
print(f'Indexed {stats[\"total_chunks\"]} chunks')
"
```

### 4. Customize the Pipeline

**Adjust retrieval settings** in `.env`:
```bash
# Number of chunks to retrieve
RETRIEVAL_TOP_K=10

# Chunk size for semantic chunking
CHUNK_SIZE=500
```

**Modify agent prompts** in:
- `src/agents/context_discovery.py`
- `src/agents/success_framework.py`

**Change scoring weights** in:
- `src/deterministic/database_explorer.py` (table scoring)
- `src/deterministic/grain_detector.py` (grain confidence)

### 5. Deploy to Production

**Option 1: FastAPI Server**
```bash
# Run with Gunicorn (production WSGI server)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

**Option 2: Serverless (AWS Lambda, Google Cloud Functions)**
- Package the `src/` directory
- Set environment variables
- Deploy with your cloud provider's CLI

**Option 3: Docker**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv sync
CMD ["uv", "run", "python", "main.py", "api"]
```

## Learning Resources

### Understanding Each Component

1. **RAG (Retrieval-Augmented Generation)**
   - Read: `src/rag/README.md` (if exists) or `docs/OBSERVABILITY_AND_EVALS.md`
   - Code: `src/rag/chunker.py`, `src/rag/vector_store.py`, `src/rag/retriever.py`
   - Key concept: Semantic search + keyword search = better retrieval

2. **LLM Agents**
   - Read: `src/agents/base_agent.py` (well-commented)
   - Code: `src/agents/context_discovery.py`, `src/agents/success_framework.py`
   - Key concept: System prompt + user prompt → structured output

3. **Deterministic Analysis**
   - Read: `src/deterministic/database_explorer.py`
   - Code: `src/deterministic/grain_detector.py`
   - Key concept: Pattern matching + heuristics = no LLM needed

4. **Observability**
   - Read: `docs/OBSERVABILITY_AND_EVALS.md`
   - Code: `src/core/observability.py`, `src/core/llm_client.py`
   - Key concept: Trace everything for debugging and improvement

### External Resources

- **Langfuse Docs**: https://langfuse.com/docs
- **Gemini API**: https://ai.google.dev/docs
- **ChromaDB**: https://docs.trychroma.com
- **FastAPI**: https://fastapi.tiangolo.com

## Troubleshooting

### Issue: "No module named 'src'"

**Solution:** Run commands with `uv run`:
```bash
uv run python main.py cli "AI Search"
```

### Issue: "API key not found"

**Solution:** Check your `.env` file:
```bash
cat .env | grep GEMINI_API_KEY
```

### Issue: "Langfuse traces not appearing"

**Solution:** Check Langfuse config:
```bash
uv run python -c "
from src.core.observability import init_observability, get_langfuse
init_observability()
print('Langfuse client:', get_langfuse())
"
```

### Issue: "Tests failing"

**Solution:** Run with verbose output:
```bash
uv run pytest evals/test_deterministic.py -v -s
```

## Summary

You now have a **production-ready AI system** with:

✅ **Multi-agent architecture** (RAG + LLM + deterministic)  
✅ **Full observability** (Langfuse tracing)  
✅ **Evaluation framework** (automated tests)  
✅ **3 deployment modes** (CLI, API, MCP)  
✅ **Clean code** (well-commented, modular)  

**Most importantly:** You understand *why* each component is built the way it is, which is the key to mastering AI engineering.

## Questions?

Check the documentation:
- `docs/OBSERVABILITY_AND_EVALS.md` - Detailed observability guide
- `README.md` - Project overview
- Code comments - Every file has detailed explanations

Or explore the codebase - it's designed to be self-documenting!

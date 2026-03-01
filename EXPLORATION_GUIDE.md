# Exploration Guide - Understanding the Agent System

This guide helps you explore each component of the system step-by-step with interactive scripts.

## 🎯 Quick Start

### 1. Explore the Vector Store (RAG)

**What you'll learn:**
- What is a vector store and why we use it
- How embeddings work (text → numbers)
- How semantic search finds relevant content
- Complete RAG (Retrieval-Augmented Generation) flow

**Run:**
```bash
uv run python explore_vector_store.py
```

**What happens:**
- Shows all chunks stored in ChromaDB
- Demonstrates how embeddings are created
- Compares similar vs different text vectors
- Runs semantic search queries
- Explains the complete RAG pipeline

**Time:** ~5-10 minutes (interactive)

---

### 2. Explore Each Agent Step-by-Step

**What you'll learn:**
- How each agent works individually
- What inputs each agent receives
- What outputs each agent produces
- How agents connect together

**Run:**
```bash
uv run python explore_agents.py
```

**What happens:**

**Step 1: Vector Store**
- Inspect what's stored in ChromaDB
- See sample chunks
- Run semantic search

**Step 2: Context Discovery Agent (RAG + LLM)**
- Search vector store for feature docs
- Send context to LLM
- Extract structured information

**Step 3: Success Framework Agent (LLM)**
- Take context from Step 2
- Generate metrics and KPIs
- Show calculation methods

**Step 4: Database Explorer (Deterministic)**
- Scan CSV files
- Extract schema information
- Score tables by relevance

**Step 5: Grain Detector (Deterministic)**
- Analyze table structure
- Detect data granularity
- Show reasoning

**Time:** ~10-15 minutes (interactive)

---

## 📊 Understanding Each Component

### Vector Store (ChromaDB)

**Location:** `src/rag/vector_store.py`

**What it does:**
- Stores document chunks as embeddings (768-dimensional vectors)
- Enables semantic search (meaning-based, not keyword-based)
- Uses Gemini embedding model

**Key concepts:**
```python
# Text to embedding
text = "search conversion metrics"
embedding = embed_text(text)  # → [0.23, -0.45, 0.12, ... 768 numbers]

# Search by similarity
results = store.search("how to measure search success", top_k=5)
# Returns chunks with similar embeddings
```

**Why vectors?**
- Similar text → Similar vectors
- Can measure similarity mathematically
- Better than keyword matching

---

### RAG (Retrieval-Augmented Generation)

**Components:**
1. **Chunker** (`src/rag/chunker.py`): Splits documents into semantic chunks
2. **Vector Store** (`src/rag/vector_store.py`): Stores chunks as embeddings
3. **Retriever** (`src/rag/retriever.py`): Finds relevant chunks
4. **LLM**: Generates answers based on retrieved context

**Flow:**
```
User Query: "What metrics for AI search?"
    ↓
Chunker: Split docs into pieces
    ↓
Embeddings: Convert chunks to vectors
    ↓
Vector Store: Store in ChromaDB
    ↓
Search: Find similar chunks
    ↓
Retrieve: Get top-K relevant chunks
    ↓
Augment: Add chunks to prompt
    ↓
LLM: Generate answer based on context
    ↓
Result: Grounded in YOUR docs
```

**Why RAG?**
- LLMs don't know about YOUR specific documents
- RAG provides context from your knowledge base
- Prevents hallucination - answers grounded in real data

---

### Context Discovery Agent

**Location:** `src/agents/context_discovery.py`

**Type:** RAG + LLM

**What it does:**
1. Searches vector store for feature documentation
2. Retrieves relevant chunks
3. Sends chunks + query to LLM
4. Extracts structured information

**Input:**
```python
feature_name = "AI Search"
additional_context = "Helps users find content faster"
```

**Output:**
```python
{
    "name": "AI Search",
    "description": "AI-powered search feature...",
    "purpose": "Help users find content faster",
    "target_users": "all users, power users",
    "success_criteria": ["increase search usage", ...],
    "recommended_metrics": ["search conversion rate", ...],
    "confidence": 0.85
}
```

**Why RAG + LLM?**
- RAG: Finds relevant docs about the feature
- LLM: Understands and structures the information

---

### Success Framework Agent

**Location:** `src/agents/success_framework.py`

**Type:** LLM only

**What it does:**
1. Takes context from Context Discovery
2. Generates specific metrics and KPIs
3. Provides calculation methods

**Input:**
```python
context = {
    "name": "AI Search",
    "purpose": "Help users find content faster",
    "target_users": "all users"
}
```

**Output:**
```python
{
    "primary_metrics": [
        {
            "display_name": "Search Conversion Rate",
            "description": "Percentage of searches leading to clicks",
            "calculation": "clicks / searches",
            "metric_type": "conversion"
        },
        ...
    ],
    "secondary_metrics": [...]
}
```

**Why LLM?**
- Requires domain knowledge (what metrics matter?)
- Needs reasoning (how to measure success?)
- Must be specific to feature type

---

### Database Explorer

**Location:** `src/deterministic/database_explorer.py`

**Type:** Deterministic (no LLM)

**What it does:**
1. Scans CSV files in data directory
2. Extracts schema (columns, types, cardinality)
3. Scores tables by keyword relevance

**Input:**
```python
data_directory = "./data/mock_warehouse"
feature_keywords = ["search", "event", "conversion"]
```

**Output:**
```python
[
    {
        "table": TableModel(...),
        "score": 0.85,
        "reason": "Matched 'search' in table name"
    },
    ...
]
```

**Why deterministic?**
- Schema introspection is pattern matching
- No reasoning needed
- Fast, free, predictable

---

### Grain Detector

**Location:** `src/deterministic/grain_detector.py`

**Type:** Deterministic (no LLM)

**What it does:**
1. Analyzes table structure
2. Looks for ID columns
3. Checks cardinality
4. Detects grain (event/session/user/firm/time)

**Input:**
```python
table = TableModel(
    name="search_events",
    columns=[
        ColumnInfoModel(name="event_id", cardinality=1000),
        ColumnInfoModel(name="user_id", cardinality=100),
        ...
    ]
)
```

**Output:**
```python
{
    "primary_grain": "event",
    "grain_column": "event_id",
    "confidence": 0.95,
    "reasoning": "High cardinality event_id column indicates event grain"
}
```

**Why deterministic?**
- Finite set of grain types
- Pattern matching on column names
- Cardinality math
- No LLM needed!

---

## 🔍 Viewing Traces in Langfuse

Every LLM call is automatically traced. Here's what to look for:

### 1. Go to Langfuse Dashboard
https://cloud.langfuse.com

### 2. View Traces
- Click **Traces** tab
- See all pipeline runs
- Each trace shows the full execution

### 3. Drill Down
Click any trace to see:
- **Timeline**: Visual representation
- **Input**: Full prompt sent to LLM
- **Output**: Complete response
- **Metadata**: Feature name, agent name, etc.
- **Latency**: How long it took
- **Cost**: Estimated API cost

### 4. View Generations
- Click **Generations** tab
- See individual LLM calls
- Filter by model, status, etc.

---

## 🧪 Running Tests

### Deterministic Tests (Fast)
```bash
# Run all deterministic tests
uv run pytest evals/test_deterministic.py -v

# Run specific test
uv run pytest evals/test_deterministic.py::TestGrainDetector::test_grain_detection -v
```

### LLM Tests (Requires API)
```bash
# Run LLM component tests
uv run pytest evals/test_llm_components.py -v

# Skip slow tests
uv run pytest evals/test_llm_components.py -m "not slow"
```

### Full Pipeline Tests
```bash
# Run end-to-end tests
uv run pytest evals/test_pipeline.py -v
```

---

## 📝 Adding Your Own Documents

### 1. Add Markdown Files
```bash
# Create a new document
cat > knowledge_base/company_docs/my_feature.md << 'EOF'
# My Feature

## Overview
Description of your feature...

## Success Metrics
- Metric 1
- Metric 2
EOF
```

### 2. Index the Knowledge Base
```bash
uv run python -c "
from src.rag.indexer import index_knowledge_base
stats = index_knowledge_base('./knowledge_base')
print(f'Indexed {stats[\"total_chunks\"]} chunks from {stats[\"total_documents\"]} documents')
"
```

### 3. Test Retrieval
```bash
uv run python -c "
from src.rag.vector_store import VectorStore
store = VectorStore(collection_name='knowledge_base')
results = store.search('your search query', top_k=3)
for r in results:
    print(f'- {r[\"metadata\"].get(\"section_title\")}: {r[\"content\"][:100]}...')
"
```

---

## 🎓 Learning Path

**Recommended order:**

1. **Run `explore_vector_store.py`**
   - Understand embeddings and semantic search
   - See RAG in action
   - ~10 minutes

2. **Run `explore_agents.py`**
   - See each agent's input/output
   - Understand the full pipeline
   - ~15 minutes

3. **Check Langfuse Dashboard**
   - View traces from the exploration
   - See prompts and responses
   - ~5 minutes

4. **Read the code**
   - Start with `src/rag/vector_store.py`
   - Then `src/agents/context_discovery.py`
   - All files have detailed comments

5. **Run tests**
   - See what gets evaluated
   - Understand quality checks
   - ~5 minutes

6. **Modify and experiment**
   - Change prompts
   - Add test cases
   - See how outputs change

---

## 🤔 Common Questions

### Q: What's the difference between semantic and keyword search?

**Keyword search:**
- Matches exact words
- "AI search" won't find "intelligent search"
- Fast but limited

**Semantic search:**
- Matches meaning
- "AI search" finds "intelligent search", "smart search", etc.
- Slower but much better

**Our solution:** Hybrid (both!)

### Q: Why not use LLM for everything?

**LLM costs:**
- Time: 2-3 seconds per call
- Money: ~$0.001 per call
- Unpredictable: Outputs can vary

**Deterministic is better when:**
- Task is pattern matching
- Output must be consistent
- Speed matters
- No reasoning needed

**Principle:** "LLM only when needed, deterministic otherwise"

### Q: How do I know if RAG is working?

**Check:**
1. Run `explore_vector_store.py`
2. Try different queries
3. See if retrieved chunks are relevant
4. Check Langfuse for retrieval traces

**Good signs:**
- High similarity scores (>0.7)
- Retrieved chunks match query intent
- LLM answers reference the chunks

### Q: Can I use a different LLM?

Yes! Edit `src/core/llm_client.py`:
- Change model name
- Update API calls
- Keep observability hooks

Supported: Gemini, OpenAI, Anthropic, etc.

---

## 📚 Additional Resources

- **Langfuse Docs**: https://langfuse.com/docs
- **ChromaDB Docs**: https://docs.trychroma.com
- **Gemini API**: https://ai.google.dev/docs
- **RAG Guide**: https://www.langfuse.com/guides/cookbook/rag

---

## ✅ Next Steps

1. Run the exploration scripts
2. Check Langfuse dashboard
3. Read the code comments
4. Add your own documents
5. Modify prompts and see changes
6. Run tests to validate

**Goal:** Understand WHY each component exists and HOW it works, so you can build your own AI systems!

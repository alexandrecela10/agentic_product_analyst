# Product Success Tracking Agent

A multi-agent system that answers: **"How do I track and quantify success for X product/feature?"**

Combines RAG-based context discovery with deterministic data analysis, following the principle: **"LLM only when needed, deterministic otherwise."**

## Features

- **Context Discovery** (RAG + LLM): Searches company docs to understand feature purpose, target users, and success criteria
- **Success Framework** (LLM): Generates recommended KPIs and tracking events
- **Database Explorer** (Deterministic): Finds relevant tables via schema introspection
- **Grain Detector** (Deterministic): Detects data granularity (event, session, user, firm, time)
- **MCP Server**: Exposes tools for AI assistants (Claude, Cursor, etc.)

## Quick Start

### 1. Install Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
~/.local/bin/uv sync
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your API keys:
# - GEMINI_API_KEY (required)
# - TAVILY_API_KEY (optional, for web search)
# - LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (optional, for observability)
```

### 3. Run the Agent

```bash
# Option 1: CLI (quick test)
~/.local/bin/uv run python main.py cli "AI Search"

# Option 2: FastAPI server
~/.local/bin/uv run python main.py api

# Option 3: MCP server (for AI assistants)
~/.local/bin/uv run python main.py mcp
```

## API Endpoints

When running the FastAPI server (`python main.py api`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/pipeline/run` | POST | Run full pipeline |
| `/rag/index` | POST | Index knowledge base |
| `/database/explore` | POST | Explore all tables |
| `/database/eligible` | POST | Find relevant tables |
| `/grain/detect` | POST | Detect table grain |

API docs available at: `http://localhost:8000/docs`

## MCP Tools

When running the MCP server (`python main.py mcp`):

| Tool | Description |
|------|-------------|
| `discover_feature_context` | RAG-powered context discovery |
| `run_full_pipeline` | Complete pipeline execution |
| `explore_database` | List all tables with metadata |
| `find_eligible_tables` | Find tables matching keywords |
| `detect_grain` | Detect table granularity |
| `index_knowledge_base` | Index documents for RAG |

## Project Structure

```
agentic_saas_analytics/
├── src/
│   ├── core/           # Config, observability, models
│   ├── rag/            # RAG pipeline (chunker, vector store, retriever)
│   ├── agents/         # LLM-backed agents
│   ├── deterministic/  # No-LLM components
│   ├── orchestrator/   # Pipeline coordination
│   ├── mcp/            # MCP server
│   └── api/            # FastAPI routes
├── knowledge_base/     # Documents for RAG
├── data/               # Mock data warehouse (CSVs)
├── tests/
└── evals/
```

## Architecture

```
User Input: "How do I track success for AI Search?"
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│           CONTEXT DISCOVERY (RAG + LLM)                 │
│  - Search company docs (PRDs, specs)                    │
│  - Extract: purpose, target users, success criteria     │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│           SUCCESS FRAMEWORK (LLM)                       │
│  - Map feature type → metric framework                  │
│  - Output: recommended KPIs, tracking events            │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│           DATABASE EXPLORER (Deterministic)             │
│  - Schema introspection, column matching                │
│  - Eligibility scoring (no LLM needed)                  │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│           GRAIN DETECTOR (Deterministic)                │
│  - Pattern matching on column names                     │
│  - Cardinality analysis                                 │
└─────────────────────────────────────────────────────────┘
```

## LLM Usage

| Component | Uses LLM? | Rationale |
|-----------|-----------|-----------|
| Context Discovery | ✅ Yes (RAG) | Requires understanding docs |
| Success Framework | ✅ Yes | Maps feature type to metrics |
| Database Explorer | ❌ No | Pure pattern matching |
| Grain Detector | ❌ No | Finite patterns, cardinality math |

**Total LLM calls per request**: ~2-3

## Development

```bash
# Install dev dependencies
~/.local/bin/uv sync --extra dev

# Run tests
~/.local/bin/uv run pytest

# Format code
~/.local/bin/uv run ruff format .

# Lint
~/.local/bin/uv run ruff check .
```

## License

MIT

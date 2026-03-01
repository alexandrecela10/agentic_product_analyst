"""
Interactive script to explore each agent step-by-step.

This script runs each component of the pipeline individually,
showing inputs and outputs at each stage so you can understand
exactly what's happening.

Run with: uv run python explore_agents.py
"""

import json
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import print as rprint

from src.core.observability import init_observability
from src.agents.context_discovery import ContextDiscoveryAgent
from src.agents.success_framework import SuccessFrameworkAgent
from src.deterministic.database_explorer import explore_database, find_eligible_tables
from src.deterministic.grain_detector import detect_grain_from_model
from src.rag.vector_store import VectorStore
from src.rag.retriever import HybridRetriever

# Initialize
console = Console()
init_observability()

def print_section(title: str):
    """Print a section header."""
    console.print(f"\n{'='*80}", style="bold blue")
    console.print(f"{title}", style="bold blue")
    console.print(f"{'='*80}\n", style="bold blue")

def print_input(label: str, content: str):
    """Print input with formatting."""
    console.print(Panel(content, title=f"📥 INPUT: {label}", border_style="green"))

def print_output(label: str, content: str):
    """Print output with formatting."""
    console.print(Panel(content, title=f"📤 OUTPUT: {label}", border_style="yellow"))

def wait_for_user():
    """Wait for user to press Enter."""
    console.print("\n[dim]Press Enter to continue...[/dim]")
    input()


# ============================================================================
# STEP 1: EXPLORE THE VECTOR STORE (RAG)
# ============================================================================

print_section("STEP 1: EXPLORING THE VECTOR STORE (RAG)")

console.print("""
[bold cyan]What is a Vector Store?[/bold cyan]

A vector store is a database that stores text as mathematical vectors (arrays of numbers).
Each piece of text is converted to a vector using an "embedding model" (we use Gemini).

[bold]Why vectors?[/bold]
- Similar text has similar vectors
- We can find relevant text by comparing vector similarity
- Much better than keyword search for understanding meaning

[bold]Our setup:[/bold]
- ChromaDB: Local vector database (no external service needed)
- Gemini embeddings: Converts text → 768-dimensional vectors
- Semantic chunking: Splits docs into meaningful pieces
""")

wait_for_user()

# Initialize vector store
console.print("\n[bold]Initializing vector store...[/bold]")
store = VectorStore(collection_name="knowledge_base")

# Show what's in the vector store
console.print("\n[bold]Checking vector store contents...[/bold]")
all_data = store.collection.get(include=["documents", "metadatas"])

if all_data and all_data['documents']:
    console.print(f"\n✅ Found {len(all_data['documents'])} chunks in vector store\n")
    
    # Show first 3 chunks
    table = Table(title="Sample Chunks in Vector Store")
    table.add_column("ID", style="cyan")
    table.add_column("Source", style="green")
    table.add_column("Section", style="yellow")
    table.add_column("Content Preview", style="white")
    
    for i in range(min(3, len(all_data['documents']))):
        doc = all_data['documents'][i]
        meta = all_data['metadatas'][i] if all_data['metadatas'] else {}
        
        table.add_row(
            all_data['ids'][i][:20] + "...",
            meta.get('document_id', 'unknown'),
            meta.get('section_title', 'N/A'),
            doc[:100] + "..." if len(doc) > 100 else doc
        )
    
    console.print(table)
else:
    console.print("\n⚠️  Vector store is empty. Run indexing first:")
    console.print("   uv run python -c \"from src.rag.indexer import index_knowledge_base; index_knowledge_base('./knowledge_base')\"")

wait_for_user()

# Demonstrate semantic search
print_section("STEP 1.1: SEMANTIC SEARCH DEMO")

console.print("""
[bold cyan]How Semantic Search Works:[/bold cyan]

1. User query: "search conversion metrics"
2. Query is converted to a vector (using Gemini)
3. Vector store finds chunks with similar vectors
4. Returns most relevant chunks ranked by similarity

Let's try it!
""")

query = "search conversion metrics"
print_input("Search Query", query)

console.print("\n[bold]Searching vector store...[/bold]")
results = store.search(query, top_k=3)

if results:
    print_output(f"Top {len(results)} Results", "")
    
    for i, result in enumerate(results, 1):
        console.print(f"\n[bold cyan]Result {i}:[/bold cyan]")
        console.print(f"  Distance: {result['distance']:.4f} (lower = more similar)")
        console.print(f"  Section: {result['metadata'].get('section_title', 'N/A')}")
        console.print(f"  Content:\n  {result['content'][:200]}...")

wait_for_user()


# ============================================================================
# STEP 2: CONTEXT DISCOVERY AGENT (RAG + LLM)
# ============================================================================

print_section("STEP 2: CONTEXT DISCOVERY AGENT")

console.print("""
[bold cyan]What does this agent do?[/bold cyan]

This agent discovers information about a feature by:
1. Searching the vector store for relevant docs (RAG)
2. Sending retrieved context + query to LLM
3. Extracting structured information (purpose, users, metrics)

[bold]Why RAG + LLM?[/bold]
- RAG: Finds relevant information from your docs
- LLM: Understands and structures the information
- Together: Accurate, grounded in your actual docs
""")

wait_for_user()

# Get user input
feature_name = console.input("\n[bold green]Enter a feature name to analyze (or press Enter for 'AI Search'): [/bold green]") or "AI Search"

print_input("Feature Name", feature_name)

console.print("\n[bold]Running Context Discovery Agent...[/bold]")
console.print("[dim]This will:[/dim]")
console.print("[dim]1. Search vector store for relevant docs[/dim]")
console.print("[dim]2. Build prompt with retrieved context[/dim]")
console.print("[dim]3. Send to Gemini LLM[/dim]")
console.print("[dim]4. Parse structured response[/dim]\n")

agent = ContextDiscoveryAgent()
context = agent.discover(feature_name)

# Show the output
print_output("Context Discovery Result", "")

result_dict = {
    "name": context.name,
    "description": context.description,
    "purpose": context.purpose,
    "target_users": context.target_users,
    "success_criteria": context.success_criteria,
    "recommended_metrics": context.recommended_metrics,
    "confidence": context.confidence
}

syntax = Syntax(json.dumps(result_dict, indent=2), "json", theme="monokai")
console.print(syntax)

console.print(f"\n[bold green]✅ Confidence Score: {context.confidence:.2f}[/bold green]")
console.print(f"[dim]This score indicates how confident the agent is about the discovered context.[/dim]")

wait_for_user()


# ============================================================================
# STEP 3: SUCCESS FRAMEWORK AGENT (LLM)
# ============================================================================

print_section("STEP 3: SUCCESS FRAMEWORK AGENT")

console.print("""
[bold cyan]What does this agent do?[/bold cyan]

This agent generates a success framework (KPIs and metrics) based on:
- The feature context from Step 2
- Best practices for metric selection
- Feature type and target users

[bold]Why LLM?[/bold]
- Requires domain knowledge (what metrics matter for search features?)
- Needs reasoning (how to measure "success"?)
- Must be specific to the feature type
""")

wait_for_user()

print_input("Feature Context", f"Using context from Step 2 for: {context.name}")

console.print("\n[bold]Running Success Framework Agent...[/bold]")
console.print("[dim]This will:[/dim]")
console.print("[dim]1. Take the context from Step 2[/dim]")
console.print("[dim]2. Build prompt asking for metrics[/dim]")
console.print("[dim]3. Send to Gemini LLM[/dim]")
console.print("[dim]4. Parse structured metric definitions[/dim]\n")

framework_agent = SuccessFrameworkAgent()
framework = framework_agent.generate(context)

# Show the output
print_output("Success Framework Result", "")

console.print(f"\n[bold]Primary Metrics ({len(framework.primary_metrics)}):[/bold]")
for i, metric in enumerate(framework.primary_metrics, 1):
    console.print(f"\n  {i}. [cyan]{metric.display_name}[/cyan]")
    console.print(f"     Type: {metric.metric_type}")
    console.print(f"     Description: {metric.description}")
    console.print(f"     Calculation: {metric.calculation}")

console.print(f"\n[bold]Secondary Metrics ({len(framework.secondary_metrics)}):[/bold]")
for i, metric in enumerate(framework.secondary_metrics, 1):
    console.print(f"\n  {i}. [yellow]{metric.display_name}[/yellow]")
    console.print(f"     Description: {metric.description[:100]}...")

wait_for_user()


# ============================================================================
# STEP 4: DATABASE EXPLORER (DETERMINISTIC)
# ============================================================================

print_section("STEP 4: DATABASE EXPLORER")

console.print("""
[bold cyan]What does this component do?[/bold cyan]

This is a DETERMINISTIC component (no LLM) that:
1. Scans CSV files in the data directory
2. Extracts schema information (columns, types, cardinality)
3. Scores tables by relevance to feature keywords

[bold]Why deterministic?[/bold]
- Schema introspection is pattern matching
- No reasoning needed, just data analysis
- Fast, free, and predictable
""")

wait_for_user()

print_input("Data Directory", "./data/mock_warehouse")

console.print("\n[bold]Exploring database...[/bold]")
tables = explore_database('./data/mock_warehouse')

print_output(f"Found {len(tables)} Tables", "")

# Show tables
table_display = Table(title="Database Tables")
table_display.add_column("Table Name", style="cyan")
table_display.add_column("Columns", style="green")
table_display.add_column("Row Count", style="yellow")
table_display.add_column("Sample Columns", style="white")

for t in tables:
    sample_cols = ", ".join([c.name for c in t.columns[:3]])
    if len(t.columns) > 3:
        sample_cols += f", ... (+{len(t.columns)-3} more)"
    
    table_display.add_row(
        t.name,
        str(len(t.columns)),
        str(t.row_count),
        sample_cols
    )

console.print(table_display)

wait_for_user()

# Find eligible tables
console.print("\n[bold]Finding tables relevant to feature...[/bold]")

# Extract keywords from context
keywords = [feature_name.lower(), "search", "event", "conversion"]
print_input("Feature Keywords", ", ".join(keywords))

eligible = find_eligible_tables(keywords, './data/mock_warehouse')

print_output(f"Found {len(eligible)} Eligible Tables", "")

# Show scored tables
scored_table = Table(title="Eligible Tables (Scored by Relevance)")
scored_table.add_column("Table Name", style="cyan")
scored_table.add_column("Score", style="green")
scored_table.add_column("Reason", style="yellow")

for scored in eligible:
    scored_table.add_row(
        scored.table.name,
        f"{scored.score:.2f}",
        f"Matched keywords in table/column names"
    )

console.print(scored_table)

wait_for_user()


# ============================================================================
# STEP 5: GRAIN DETECTOR (DETERMINISTIC)
# ============================================================================

print_section("STEP 5: GRAIN DETECTOR")

console.print("""
[bold cyan]What is "grain"?[/bold cyan]

Grain is the level of detail in a table:
- EVENT grain: One row per event (search_events, clicks)
- SESSION grain: One row per session
- USER grain: One row per user
- FIRM grain: One row per company
- TIME grain: One row per time period

[bold]Why does grain matter?[/bold]
- Determines how to aggregate metrics
- Affects what calculations are possible
- Critical for correct metric definitions

[bold]How we detect it (deterministically):[/bold]
1. Look for ID columns (event_id, session_id, user_id)
2. Check cardinality (how many unique values)
3. Apply heuristics (high cardinality = event grain)
4. No LLM needed - pure pattern matching!
""")

wait_for_user()

console.print("\n[bold]Detecting grain for each table...[/bold]")

grain_table = Table(title="Grain Detection Results")
grain_table.add_column("Table", style="cyan")
grain_table.add_column("Grain", style="green")
grain_table.add_column("Grain Column", style="yellow")
grain_table.add_column("Confidence", style="magenta")
grain_table.add_column("Reasoning", style="white")

for scored in eligible[:3]:  # Just first 3 for demo
    print_input(f"Analyzing Table", scored.table.name)
    
    grain_result = detect_grain_from_model(scored.table)
    
    grain_table.add_row(
        scored.table.name,
        grain_result.primary_grain.value,
        grain_result.grain_column,
        f"{grain_result.confidence:.2f}",
        grain_result.reasoning[:50] + "..."
    )

console.print(grain_table)

wait_for_user()


# ============================================================================
# SUMMARY
# ============================================================================

print_section("SUMMARY: COMPLETE PIPELINE")

console.print("""
[bold cyan]You just ran the complete pipeline step-by-step![/bold cyan]

Here's what happened:

[bold green]1. Vector Store (RAG)[/bold green]
   - Searched knowledge base for relevant docs
   - Used semantic similarity (vectors) not keywords
   - Found context about the feature

[bold green]2. Context Discovery Agent (RAG + LLM)[/bold green]
   - Retrieved relevant chunks from vector store
   - Sent to LLM with structured prompt
   - Extracted feature purpose, users, criteria

[bold green]3. Success Framework Agent (LLM)[/bold green]
   - Took context from step 2
   - Generated specific metrics and KPIs
   - Provided calculation methods

[bold green]4. Database Explorer (Deterministic)[/bold green]
   - Scanned CSV files
   - Extracted schema information
   - Scored tables by relevance

[bold green]5. Grain Detector (Deterministic)[/bold green]
   - Analyzed table structure
   - Detected data granularity
   - No LLM needed - pure logic!

[bold yellow]Key Principle:[/bold yellow]
"LLM only when needed, deterministic otherwise"

- Steps 1-3: Need understanding → Use LLM
- Steps 4-5: Pure pattern matching → Deterministic

[bold cyan]Check Langfuse Dashboard:[/bold cyan]
https://cloud.langfuse.com

You'll see traces for:
- Context discovery LLM call
- Success framework LLM call
- Full inputs and outputs
""")

console.print("\n[bold green]✅ Exploration complete![/bold green]\n")

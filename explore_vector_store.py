"""
Deep dive into the vector store to understand RAG.

This script shows you exactly what's stored in the vector database,
how embeddings work, and how semantic search finds relevant content.

Run with: uv run python explore_vector_store.py
"""

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from src.rag.vector_store import VectorStore
from src.core.llm_client import embed_text, embed_query

console = Console()

def print_section(title: str):
    console.print(f"\n{'='*80}", style="bold blue")
    console.print(f"{title}", style="bold blue")
    console.print(f"{'='*80}\n", style="bold blue")


print_section("UNDERSTANDING THE VECTOR STORE")

console.print("""
[bold cyan]What is a Vector Store?[/bold cyan]

Think of it as a special database that stores:
1. [bold]Text chunks[/bold]: Pieces of your documents
2. [bold]Embeddings[/bold]: Mathematical representations (vectors) of that text
3. [bold]Metadata[/bold]: Information about where the text came from

[bold]Why embeddings?[/bold]
- Text: "AI search feature" → Vector: [0.23, -0.45, 0.12, ... 768 numbers]
- Similar text → Similar vectors
- We can measure similarity mathematically

Let's explore!
""")

input("\nPress Enter to continue...")


# ============================================================================
# PART 1: WHAT'S IN THE VECTOR STORE?
# ============================================================================

print_section("PART 1: INSPECTING VECTOR STORE CONTENTS")

console.print("[bold]Connecting to vector store...[/bold]")
store = VectorStore(collection_name="knowledge_base")

# Get all data
all_data = store.collection.get(
    include=["documents", "metadatas", "embeddings"]
)

if not all_data or not all_data['documents']:
    console.print("\n[red]⚠️  Vector store is empty![/red]")
    console.print("\nRun this to index your knowledge base:")
    console.print("[cyan]uv run python -c \"from src.rag.indexer import index_knowledge_base; index_knowledge_base('./knowledge_base')\"[/cyan]")
    exit(1)

num_chunks = len(all_data['documents'])
console.print(f"\n[green]✅ Found {num_chunks} chunks in the vector store[/green]\n")

# Show statistics
console.print("[bold]Vector Store Statistics:[/bold]")
console.print(f"  Total chunks: {num_chunks}")
console.print(f"  Embedding dimensions: {len(all_data['embeddings'][0]) if all_data['embeddings'] else 'N/A'}")

# Count by document
doc_counts = {}
for meta in all_data['metadatas']:
    doc_id = meta.get('document_id', 'unknown')
    doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

console.print(f"  Unique documents: {len(doc_counts)}")
console.print("\n[bold]Chunks per document:[/bold]")
for doc_id, count in doc_counts.items():
    console.print(f"  - {doc_id}: {count} chunks")

input("\nPress Enter to see sample chunks...")


# ============================================================================
# PART 2: VIEWING ACTUAL CHUNKS
# ============================================================================

print_section("PART 2: SAMPLE CHUNKS")

console.print("[bold]Here are 5 random chunks from your knowledge base:[/bold]\n")

# Show 5 sample chunks
for i in range(min(5, num_chunks)):
    doc = all_data['documents'][i]
    meta = all_data['metadatas'][i]
    
    console.print(f"\n[bold cyan]Chunk {i+1}:[/bold cyan]")
    console.print(f"  ID: {all_data['ids'][i]}")
    console.print(f"  Source: {meta.get('document_id', 'unknown')}")
    console.print(f"  Section: {meta.get('section_title', 'N/A')}")
    console.print(f"  Length: {len(doc)} characters")
    console.print(f"\n  Content:")
    console.print(Panel(doc[:300] + ("..." if len(doc) > 300 else ""), border_style="dim"))

input("\nPress Enter to understand embeddings...")


# ============================================================================
# PART 3: UNDERSTANDING EMBEDDINGS
# ============================================================================

print_section("PART 3: WHAT ARE EMBEDDINGS?")

console.print("""
[bold cyan]Embeddings are vectors (arrays of numbers) that represent text.[/bold cyan]

[bold]Example:[/bold]
Text: "AI search helps users find content"
Embedding: [0.234, -0.456, 0.123, 0.789, ... 768 numbers total]

[bold]Key properties:[/bold]
1. Similar text → Similar vectors
2. We can measure similarity using math (cosine similarity, distance)
3. This is how semantic search works!

Let's see a real embedding:
""")

# Show a real embedding
sample_embedding = all_data['embeddings'][0]
console.print(f"\n[bold]Sample embedding (first 10 dimensions):[/bold]")
console.print(f"  {sample_embedding[:10]}")
console.print(f"\n[dim]Full embedding has {len(sample_embedding)} dimensions[/dim]")

input("\nPress Enter to see how embeddings are created...")


# ============================================================================
# PART 4: CREATING EMBEDDINGS
# ============================================================================

print_section("PART 4: CREATING EMBEDDINGS")

console.print("""
[bold cyan]How we create embeddings:[/bold cyan]

1. Take text: "search conversion metrics"
2. Send to Gemini embedding model
3. Get back a 768-dimensional vector
4. Store in ChromaDB

Let's create an embedding live!
""")

test_text = "search conversion metrics"
console.print(f"\n[bold]Input text:[/bold] \"{test_text}\"")
console.print("\n[dim]Calling Gemini embedding API...[/dim]")

embedding = embed_query(test_text)

console.print(f"\n[green]✅ Created embedding![/green]")
console.print(f"  Dimensions: {len(embedding)}")
console.print(f"  First 10 values: {embedding[:10]}")
console.print(f"  Data type: {type(embedding[0])}")

# Show that similar text has similar embeddings
console.print("\n[bold]Let's compare similar vs different text:[/bold]")

text1 = "search conversion rate"
text2 = "search click-through rate"  # Similar
text3 = "user authentication system"  # Different

console.print(f"\n[dim]Creating embeddings for comparison...[/dim]")
emb1 = np.array(embed_query(text1))
emb2 = np.array(embed_query(text2))
emb3 = np.array(embed_query(text3))

# Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_1_2 = cosine_similarity(emb1, emb2)
sim_1_3 = cosine_similarity(emb1, emb3)

console.print(f"\n[bold]Similarity scores (higher = more similar):[/bold]")
console.print(f"  \"{text1}\" vs \"{text2}\": [green]{sim_1_2:.4f}[/green]")
console.print(f"  \"{text1}\" vs \"{text3}\": [yellow]{sim_1_3:.4f}[/yellow]")
console.print(f"\n[dim]Notice: Similar topics have higher similarity![/dim]")

input("\nPress Enter to see semantic search in action...")


# ============================================================================
# PART 5: SEMANTIC SEARCH
# ============================================================================

print_section("PART 5: SEMANTIC SEARCH IN ACTION")

console.print("""
[bold cyan]How semantic search works:[/bold cyan]

1. User query: "how to measure search success"
2. Convert query to embedding (vector)
3. Find chunks with similar embeddings
4. Return most similar chunks

This finds relevant content even if exact words don't match!
""")

# Try different queries
queries = [
    "search conversion metrics",
    "how to measure search success",
    "user engagement tracking",
]

for query in queries:
    console.print(f"\n[bold cyan]Query:[/bold cyan] \"{query}\"")
    console.print("[dim]Searching...[/dim]")
    
    results = store.search(query, top_k=3)
    
    console.print(f"\n[bold]Top 3 results:[/bold]")
    
    for i, result in enumerate(results, 1):
        distance = result['distance']
        similarity = 1 - distance  # Convert distance to similarity
        
        console.print(f"\n  {i}. [green]Similarity: {similarity:.4f}[/green]")
        console.print(f"     Section: {result['metadata'].get('section_title', 'N/A')}")
        console.print(f"     Preview: {result['content'][:100]}...")
    
    input("\nPress Enter for next query...")


# ============================================================================
# PART 6: RAG EXPLAINED
# ============================================================================

print_section("PART 6: HOW RAG WORKS")

console.print("""
[bold cyan]RAG = Retrieval-Augmented Generation[/bold cyan]

[bold]The problem RAG solves:[/bold]
- LLMs don't know about YOUR specific documents
- LLMs can hallucinate (make things up)
- LLMs have a knowledge cutoff date

[bold]How RAG fixes this:[/bold]

1. [bold green]RETRIEVAL[/bold green] (what we just explored)
   - User asks: "What metrics for AI search?"
   - Search vector store for relevant chunks
   - Find: PRD sections about search metrics
   
2. [bold yellow]AUGMENTATION[/bold yellow]
   - Take retrieved chunks
   - Add them to the LLM prompt
   - "Here's context from our docs: [chunks]"
   
3. [bold blue]GENERATION[/bold blue]
   - LLM reads the context
   - Generates answer based on YOUR docs
   - No hallucination - grounded in real data

[bold]Our RAG pipeline:[/bold]

User Query
    ↓
Vector Store Search (semantic)
    ↓
Retrieve Top-K Chunks
    ↓
Build Prompt: "Context: [chunks]\nQuestion: [query]"
    ↓
Send to Gemini LLM
    ↓
Get Grounded Answer
""")

input("\nPress Enter to see a complete RAG example...")


# ============================================================================
# PART 7: COMPLETE RAG EXAMPLE
# ============================================================================

print_section("PART 7: COMPLETE RAG EXAMPLE")

console.print("[bold]Let's run a complete RAG query:[/bold]\n")

query = "What are the key metrics for a search feature?"
console.print(f"[bold cyan]Query:[/bold cyan] {query}")

# Step 1: Retrieve
console.print("\n[bold green]Step 1: RETRIEVAL[/bold green]")
console.print("[dim]Searching vector store...[/dim]")
results = store.search(query, top_k=3)

console.print(f"\n✅ Retrieved {len(results)} relevant chunks:")
for i, result in enumerate(results, 1):
    console.print(f"\n  Chunk {i}:")
    console.print(f"  Source: {result['metadata'].get('document_id', 'unknown')}")
    console.print(f"  Section: {result['metadata'].get('section_title', 'N/A')}")
    console.print(f"  Content: {result['content'][:150]}...")

# Step 2: Augment
console.print("\n[bold yellow]Step 2: AUGMENTATION[/bold yellow]")
console.print("[dim]Building prompt with retrieved context...[/dim]")

context_text = "\n\n".join([r['content'] for r in results])
augmented_prompt = f"""
Based on the following context from our documentation:

{context_text}

Question: {query}

Please provide a specific answer based on the context above.
"""

console.print("\n✅ Prompt built with context")
console.print(Panel(augmented_prompt[:500] + "...", title="Augmented Prompt", border_style="yellow"))

# Step 3: Generate
console.print("\n[bold blue]Step 3: GENERATION[/bold blue]")
console.print("[dim]This is where we'd send to LLM...[/dim]")
console.print("[dim](Skipping actual LLM call to save API costs)[/dim]")

console.print("""
\n[bold]What would happen:[/bold]
1. Prompt sent to Gemini
2. LLM reads the context
3. Generates answer based on YOUR docs
4. Returns structured response

[bold green]Result: Accurate answer grounded in your documentation![/bold green]
""")


# ============================================================================
# SUMMARY
# ============================================================================

print_section("SUMMARY: VECTOR STORE & RAG")

console.print("""
[bold cyan]What you learned:[/bold cyan]

[bold]1. Vector Store[/bold]
   ✅ Stores text as mathematical vectors (embeddings)
   ✅ Uses Gemini to create 768-dimensional embeddings
   ✅ Enables semantic search (meaning-based, not keyword-based)

[bold]2. Embeddings[/bold]
   ✅ Convert text to numbers
   ✅ Similar text → Similar vectors
   ✅ Measured using cosine similarity

[bold]3. Semantic Search[/bold]
   ✅ Query → Embedding → Find similar embeddings
   ✅ Returns relevant chunks even without exact word matches
   ✅ Much better than keyword search

[bold]4. RAG (Retrieval-Augmented Generation)[/bold]
   ✅ Retrieval: Find relevant chunks from vector store
   ✅ Augmentation: Add chunks to LLM prompt
   ✅ Generation: LLM generates answer based on YOUR docs

[bold cyan]Key files to explore:[/bold cyan]
- src/rag/vector_store.py - Vector store wrapper
- src/rag/chunker.py - How we split documents
- src/rag/retriever.py - Hybrid search (semantic + keyword)
- src/rag/indexer.py - How we index documents

[bold cyan]Try this next:[/bold cyan]
1. Run: uv run python explore_agents.py
   → See how RAG is used in the full pipeline

2. Check Langfuse dashboard
   → See embeddings being created in real-time

3. Add your own documents to knowledge_base/
   → Re-index and search your own content
""")

console.print("\n[bold green]✅ Vector store exploration complete![/bold green]\n")

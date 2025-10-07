from src.embedding_system import EmbeddingSystem
from src.retrieval_pipeline import RetrievalPipeline

print("="*70)
print("TESTING RETRIEVAL SYSTEM")
print("="*70)

# Initialize
print("\n📥 Loading indices...")
embedding_sys = EmbeddingSystem()
embedding_sys.load_indices()

retrieval = RetrievalPipeline(embedding_sys)
print("✓ System loaded successfully")

# Test queries
test_queries = [
    "What is the maximum deduction limit under Section 80C?",
    "How to calculate HRA exemption?",
    "What are the tax slabs for individual taxpayers?",
    "Explain Section 10 exemptions"
]

for query in test_queries:
    print("\n" + "="*70)
    print(f"🔍 Query: {query}")
    print("="*70)
    
    result = retrieval.retrieve(query, top_k=3)
    
    print(f"\nQuery Type: {result['query_metadata']['query_type']}")
    print(f"Results Found: {result['num_results']}")
    
    for i, (chunk, score) in enumerate(zip(result['chunks'], result['scores']), 1):
        print(f"\n  [{i}] Score: {score:.3f}")
        print(f"      Section: {chunk['section_number']} - {chunk['section_title']}")
        print(f"      Preview: {chunk['text'][:150]}...")

print("\n" + "="*70)
print("✅ RETRIEVAL TEST COMPLETE")
print("="*70)
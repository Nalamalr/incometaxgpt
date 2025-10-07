from src.embedding_system import EmbeddingSystem
import time

print("="*70)
print("BUILDING RETRIEVAL INDICES")
print("="*70)
print("\nThis will take 10-30 minutes on first run...")
print("The system will:")
print("1. Load processed documents")
print("2. Generate embeddings using sentence-transformers")
print("3. Build FAISS index for semantic search")
print("4. Build BM25 index for lexical search")
print("\n" + "="*70)

start_time = time.time()

# Initialize
embedding_sys = EmbeddingSystem()

# Build all indices
embedding_sys.build_all_indices(force_rebuild=True)

elapsed = time.time() - start_time

print("\n" + "="*70)
print("✅ INDEX BUILDING COMPLETE!")
print("="*70)
print(f"Time taken: {elapsed/60:.1f} minutes")
print(f"\nGenerated files:")
print("  - data/embeddings/embeddings.pkl")
print("  - data/embeddings/faiss.index")
print("  - data/embeddings/bm25.pkl")
print("  - data/embeddings/chunks.json")
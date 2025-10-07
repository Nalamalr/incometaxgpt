"""
Embedding and Vector Database System for IncomeTaxGPT
Handles semantic search using FAISS and sentence transformers
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

class EmbeddingSystem:
    def __init__(self, 
                 processed_dir: str = "data/processed",
                 embeddings_dir: str = "data/embeddings",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding system
        
        Args:
            processed_dir: Directory with processed chunks
            embeddings_dir: Directory to store embeddings
            model_name: HuggingFace model for embeddings (free, small, fast)
        """
        self.processed_dir = Path(processed_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Storage
        self.chunks = []
        self.index = None
        self.bm25 = None
    
    def load_all_chunks(self) -> List[Dict]:
        """Load all processed chunks"""
        print("\n📚 Loading processed chunks...")
        all_chunks = []
        
        for chunk_file in self.processed_dir.glob("*_chunks.json"):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
        
        print(f"Loaded {len(all_chunks)} chunks")
        return all_chunks
    
    def create_embeddings(self, force_rebuild: bool = False):
        """
        Create embeddings for all chunks
        """
        embeddings_file = self.embeddings_dir / "embeddings.pkl"
        chunks_file = self.embeddings_dir / "chunks.json"
        
        # Load existing if available
        if not force_rebuild and embeddings_file.exists():
            print("Loading existing embeddings...")
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            print(f"Loaded {len(self.chunks)} chunks with embeddings")
            return embeddings
        
        # Create new embeddings
        print("\n🔄 Creating new embeddings...")
        self.chunks = self.load_all_chunks()
        
        if not self.chunks:
            raise ValueError("No chunks found! Process documents first.")
        
        # Extract text for encoding
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encoder.encode(batch, 
                                                   show_progress_bar=False,
                                                   convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        # Save embeddings
        print("Saving embeddings...")
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created embeddings: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index for fast similarity search"""
        print("\n🏗️  Building FAISS index...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index (using simple flat index for < 1M documents)
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product = cosine after normalization
        self.index.add(embeddings)
        
        # Save index
        index_file = self.embeddings_dir / "faiss.index"
        faiss.write_index(self.index, str(index_file))
        
        print(f"✓ FAISS index built: {self.index.ntotal} vectors")
    
    def build_bm25_index(self):
        """Build BM25 index for lexical search"""
        print("\n📝 Building BM25 index...")
        
        # Tokenize texts
        tokenized_corpus = [chunk['text'].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Save BM25
        bm25_file = self.embeddings_dir / "bm25.pkl"
        with open(bm25_file, 'wb') as f:
            pickle.dump(self.bm25, f)
        
        print(f"✓ BM25 index built")
    
    def build_all_indices(self, force_rebuild: bool = False):
        """Build all retrieval indices"""
        # Create embeddings
        embeddings = self.create_embeddings(force_rebuild)
        
        # Build FAISS index
        self.build_faiss_index(embeddings)
        
        # Build BM25 index
        self.build_bm25_index()
        
        print("\n✅ All indices built successfully!")
    
    def load_indices(self):
        """Load pre-built indices"""
        print("\n📥 Loading indices...")
        
        # Load chunks
        chunks_file = self.embeddings_dir / "chunks.json"
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # Load FAISS index
        index_file = self.embeddings_dir / "faiss.index"
        self.index = faiss.read_index(str(index_file))
        
        # Load BM25
        bm25_file = self.embeddings_dir / "bm25.pkl"
        with open(bm25_file, 'rb') as f:
            self.bm25 = pickle.load(f)
        
        print(f"✓ Loaded indices: {len(self.chunks)} chunks")
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Perform semantic search using FAISS
        
        Returns: List of (chunk, score) tuples
        """
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Return chunks with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append((self.chunks[idx], float(score)))
        
        return results
    
    def lexical_search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Perform lexical search using BM25
        
        Returns: List of (chunk, score) tuples
        """
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return chunks with scores
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))
        
        return results
    
    def hybrid_search(self, 
                     query: str, 
                     top_k: int = 5,
                     semantic_weight: float = 0.7) -> List[Tuple[Dict, float]]:
        """
        Hybrid search combining semantic and lexical
        
        Args:
            query: Search query
            top_k: Number of results
            semantic_weight: Weight for semantic search (0-1)
        """
        # Get more results from each method
        k_per_method = top_k * 2
        
        # Semantic search
        semantic_results = self.semantic_search(query, k_per_method)
        
        # Lexical search
        lexical_results = self.lexical_search(query, k_per_method)
        
        # Combine scores
        combined_scores = {}
        
        # Add semantic scores
        for chunk, score in semantic_results:
            chunk_id = chunk['chunk_id']
            combined_scores[chunk_id] = {
                'chunk': chunk,
                'score': score * semantic_weight
            }
        
        # Add lexical scores
        for chunk, score in lexical_results:
            chunk_id = chunk['chunk_id']
            if chunk_id in combined_scores:
                combined_scores[chunk_id]['score'] += score * (1 - semantic_weight)
            else:
                combined_scores[chunk_id] = {
                    'chunk': chunk,
                    'score': score * (1 - semantic_weight)
                }
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.values(), 
                              key=lambda x: x['score'], 
                              reverse=True)
        
        # Return top-k
        return [(item['chunk'], item['score']) for item in sorted_results[:top_k]]

# Example usage
if __name__ == "__main__":
    # Initialize system
    embedding_sys = EmbeddingSystem()
    
    # Build indices (only needed once)
    embedding_sys.build_all_indices(force_rebuild=False)
    
    # Load indices for searching
    embedding_sys.load_indices()
    
    # Test queries
    test_queries = [
        "What is the limit for Section 80C deductions?",
        "How to calculate HRA exemption?",
        "Tax rates for old regime"
    ]
    
    print("\n" + "="*50)
    print("TESTING HYBRID SEARCH")
    print("="*50)
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = embedding_sys.hybrid_search(query, top_k=3)
        
        for i, (chunk, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   Section: {chunk['section_number']} - {chunk['section_title']}")
            print(f"   Text: {chunk['text'][:150]}...")
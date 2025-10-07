"""
Advanced Retrieval Pipeline for IncomeTaxGPT
Handles query processing, filtering, and context preparation
"""

from typing import List, Dict, Tuple, Optional
from datetime import datetime
import re

class RetrievalPipeline:
    def __init__(self, embedding_system):
        """
        Initialize retrieval pipeline
        
        Args:
            embedding_system: Initialized EmbeddingSystem instance
        """
        self.embedding_sys = embedding_system
        
        # Query processing patterns
        self.year_pattern = r'\b(20\d{2}|AY\s*20\d{2}-\d{2}|FY\s*20\d{2}-\d{2})\b'
        self.section_pattern = r'section\s*(\d+[A-Z]*)'
        
    def extract_query_metadata(self, query: str) -> Dict:
        """
        Extract metadata from query (year, section references, etc.)
        """
        metadata = {
            'year': None,
            'sections': [],
            'query_type': self._classify_query_type(query)
        }
        
        # Extract year
        year_match = re.search(self.year_pattern, query, re.IGNORECASE)
        if year_match:
            year_str = year_match.group(1)
            # Convert to int (handle AY/FY formats)
            if 'AY' in year_str or 'FY' in year_str:
                year = int(re.search(r'20\d{2}', year_str).group())
            else:
                year = int(year_str)
            metadata['year'] = year
        
        # Extract section numbers
        section_matches = re.findall(self.section_pattern, query, re.IGNORECASE)
        metadata['sections'] = section_matches
        
        return metadata
    
    def _classify_query_type(self, query: str) -> str:
        """
        Classify query type for better retrieval
        """
        query_lower = query.lower()
        
        # Keywords for different query types
        calculation_keywords = ['calculate', 'computation', 'how much', 'amount', 'tax liability']
        explanation_keywords = ['what is', 'explain', 'define', 'meaning', 'eligibility']
        procedure_keywords = ['how to', 'process', 'steps', 'procedure', 'file', 'claim']
        comparison_keywords = ['difference', 'compare', 'vs', 'versus', 'better']
        
        if any(kw in query_lower for kw in calculation_keywords):
            return 'calculation'
        elif any(kw in query_lower for kw in explanation_keywords):
            return 'explanation'
        elif any(kw in query_lower for kw in procedure_keywords):
            return 'procedure'
        elif any(kw in query_lower for kw in comparison_keywords):
            return 'comparison'
        else:
            return 'general'
    
    def filter_by_year(self, chunks: List[Dict], target_year: int) -> List[Dict]:
        """
        Filter chunks by year relevance
        """
        # Prefer chunks from target year or latest if not available
        filtered = [c for c in chunks if c.get('year') == target_year]
        
        if not filtered:
            # Fall back to latest year
            latest_year = max(c.get('year', 0) for c in chunks)
            filtered = [c for c in chunks if c.get('year') == latest_year]
        
        return filtered
    
    def filter_by_section(self, chunks: List[Dict], section_numbers: List[str]) -> List[Dict]:
        """
        Filter chunks by section numbers
        """
        if not section_numbers:
            return chunks
        
        filtered = []
        for chunk in chunks:
            chunk_section = chunk.get('section_number', '').upper()
            if any(sec.upper() in chunk_section for sec in section_numbers):
                filtered.append(chunk)
        
        return filtered if filtered else chunks
    
    def rerank_results(self, 
                      chunks: List[Tuple[Dict, float]], 
                      query_metadata: Dict) -> List[Tuple[Dict, float]]:
        """
        Rerank results based on query metadata and relevance signals
        """
        reranked = []
        
        for chunk, score in chunks:
            bonus = 0.0
            
            # Boost if section matches
            if query_metadata['sections']:
                chunk_section = chunk.get('section_number', '').upper()
                if any(sec.upper() in chunk_section for sec in query_metadata['sections']):
                    bonus += 0.2
            
            # Boost if year matches
            if query_metadata['year']:
                if chunk.get('year') == query_metadata['year']:
                    bonus += 0.1
            
            # Boost if doc type is primary (Act vs Circular)
            if chunk.get('doc_type') == 'act':
                bonus += 0.05
            
            reranked.append((chunk, score + bonus))
        
        # Sort by adjusted score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    
    def prepare_context(self, 
                       chunks: List[Tuple[Dict, float]], 
                       max_tokens: int = 2000) -> str:
        """
        Prepare context string from chunks for LLM
        
        Args:
            chunks: List of (chunk, score) tuples
            max_tokens: Maximum context length in tokens (approx 4 chars = 1 token)
        """
        max_chars = max_tokens * 4
        context_parts = []
        current_length = 0
        
        for i, (chunk, score) in enumerate(chunks):
            # Format chunk with metadata
            chunk_text = f"""
[Source {i+1}]
Document: {chunk['doc_title']}
Section: {chunk['section_number']} - {chunk['section_title']}
Year: {chunk['year']}

{chunk['text']}

---
"""
            chunk_length = len(chunk_text)
            
            if current_length + chunk_length > max_chars:
                break
            
            context_parts.append(chunk_text)
            current_length += chunk_length
        
        return "\n".join(context_parts)
    
    def retrieve(self, 
                query: str,
                top_k: int = 5,
                include_context: bool = True) -> Dict:
        """
        Main retrieval function
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            include_context: Whether to prepare formatted context
        
        Returns:
            Dictionary with retrieved chunks, metadata, and formatted context
        """
        # Extract query metadata
        query_metadata = self.extract_query_metadata(query)
        
        # Perform hybrid search
        results = self.embedding_sys.hybrid_search(query, top_k=top_k * 2)
        
        # Apply filters
        if query_metadata['year']:
            filtered_chunks = self.filter_by_year(
                [c for c, _ in results], 
                query_metadata['year']
            )
            results = [(c, s) for c, s in results if c in filtered_chunks]
        
        if query_metadata['sections']:
            filtered_chunks = self.filter_by_section(
                [c for c, _ in results],
                query_metadata['sections']
            )
            results = [(c, s) for c, s in results if c in filtered_chunks]
        
        # Rerank
        results = self.rerank_results(results, query_metadata)
        
        # Take top-k after reranking
        results = results[:top_k]
        
        # Prepare response
        response = {
            'query': query,
            'query_metadata': query_metadata,
            'chunks': [c for c, _ in results],
            'scores': [s for _, s in results],
            'num_results': len(results)
        }
        
        # Add formatted context if requested
        if include_context:
            response['context'] = self.prepare_context(results)
        
        return response
    
    def get_section_details(self, section_number: str) -> Optional[Dict]:
        """
        Get all chunks for a specific section
        """
        section_chunks = [
            c for c in self.embedding_sys.chunks 
            if c.get('section_number', '').upper() == section_number.upper()
        ]
        
        if not section_chunks:
            return None
        
        # Sort by chunk index
        section_chunks.sort(key=lambda x: x['metadata'].get('chunk_index', 0))
        
        return {
            'section_number': section_number,
            'chunks': section_chunks,
            'full_text': '\n\n'.join(c['text'] for c in section_chunks)
        }

# Example usage
if __name__ == "__main__":
    from embedding_system import EmbeddingSystem
    
    # Initialize systems
    embedding_sys = EmbeddingSystem()
    embedding_sys.load_indices()
    
    retrieval = RetrievalPipeline(embedding_sys)
    
    # Test queries
    test_queries = [
        "What is the maximum deduction limit under Section 80C for FY 2023-24?",
        "How to calculate HRA exemption?",
        "Explain the difference between old and new tax regime",
        "Section 10 exemptions available to salaried employees"
    ]
    
    print("\n" + "="*70)
    print("TESTING ADVANCED RETRIEVAL PIPELINE")
    print("="*70)
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print("="*70)
        
        result = retrieval.retrieve(query, top_k=3)
        
        print(f"\nQuery Type: {result['query_metadata']['query_type']}")
        print(f"Detected Year: {result['query_metadata']['year']}")
        print(f"Detected Sections: {result['query_metadata']['sections']}")
        print(f"Results Found: {result['num_results']}")
        
        print("\n--- Retrieved Chunks ---")
        for i, (chunk, score) in enumerate(zip(result['chunks'], result['scores']), 1):
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   Section: {chunk['section_number']} - {chunk['section_title']}")
            print(f"   Year: {chunk['year']}")
            print(f"   Preview: {chunk['text'][:200]}...")
        
        print("\n--- Formatted Context (First 500 chars) ---")
        print(result['context'][:500] + "...")
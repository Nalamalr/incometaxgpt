"""
Document Processing Pipeline for IncomeTaxGPT
Extracts, chunks, and structures legal documents
FIXED VERSION - UTF-8 encoding for all file operations
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
import json
from pypdf import PdfReader
from dataclasses import dataclass, asdict
from tqdm import tqdm

@dataclass
class LegalChunk:
    """Structured chunk of legal text"""
    chunk_id: str
    doc_id: str
    doc_type: str
    doc_title: str
    section_number: str
    section_title: str
    text: str
    year: int
    metadata: Dict
    
    def to_dict(self):
        return asdict(self)

class DocumentProcessor:
    def __init__(self, raw_dir: str = "data/raw", 
                 processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Chunk settings
        self.max_chunk_length = 512  # tokens (roughly 400 words)
        self.chunk_overlap = 50
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return ""
    
    def identify_sections(self, text: str) -> List[Dict]:
        """
        Identify sections in legal text
        Pattern: Section XXX. Title
        """
        # Pattern for sections in Income Tax Act
        section_pattern = r'(Section\s+(\d+[A-Z]*))\.\s*([^\n]+)'
        
        sections = []
        matches = re.finditer(section_pattern, text, re.IGNORECASE)
        
        prev_end = 0
        prev_section = None
        
        for match in matches:
            section_num = match.group(2)
            section_title = match.group(3).strip()
            section_start = match.start()
            
            # Save previous section's content
            if prev_section:
                prev_section['text'] = text[prev_end:section_start].strip()
                sections.append(prev_section)
            
            # Start new section
            prev_section = {
                'section_number': section_num,
                'section_title': section_title,
                'start_pos': section_start
            }
            prev_end = match.end()
        
        # Add last section
        if prev_section:
            prev_section['text'] = text[prev_end:].strip()
            sections.append(prev_section)
        
        return sections
    
    def chunk_text(self, text: str, max_length: int = 512) -> List[str]:
        """
        Split text into chunks with overlap
        max_length is in characters (roughly 512 chars ~ 100 words)
        """
        words = text.split()
        chunks = []
        
        # Approximate: 5 chars per word
        words_per_chunk = max_length // 5
        overlap_words = self.chunk_overlap // 5
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + words_per_chunk]
            chunks.append(' '.join(chunk_words))
            i += words_per_chunk - overlap_words
        
        return chunks
    
    def process_document(self, 
                        doc_path: str,
                        doc_id: str,
                        doc_type: str,
                        doc_title: str,
                        year: int) -> List[LegalChunk]:
        """
        Process a legal document into structured chunks
        """
        print(f"\n📄 Processing: {doc_title}")
        
        # Extract text
        if doc_path.endswith('.pdf'):
            text = self.extract_text_from_pdf(doc_path)
        else:
            with open(doc_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        if not text:
            print("⚠️  No text extracted!")
            return []
        
        # Identify sections
        sections = self.identify_sections(text)
        print(f"Found {len(sections)} sections")
        
        # Create chunks
        all_chunks = []
        chunk_counter = 0
        
        for section in tqdm(sections, desc="Creating chunks"):
            section_text = section.get('text', '')
            if len(section_text) < 50:  # Skip very short sections
                continue
            
            # Split long sections into chunks
            text_chunks = self.chunk_text(section_text, self.max_chunk_length)
            
            for idx, chunk_text in enumerate(text_chunks):
                chunk = LegalChunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_counter}",
                    doc_id=doc_id,
                    doc_type=doc_type,
                    doc_title=doc_title,
                    section_number=section.get('section_number', 'unknown'),
                    section_title=section.get('section_title', 'Unknown'),
                    text=chunk_text,
                    year=year,
                    metadata={
                        'chunk_index': idx,
                        'total_chunks_in_section': len(text_chunks),
                        'section_length': len(section_text)
                    }
                )
                all_chunks.append(chunk)
                chunk_counter += 1
        
        print(f"✓ Created {len(all_chunks)} chunks")
        
        # Save processed chunks with UTF-8 encoding
        output_file = self.processed_dir / f"{doc_id}_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([c.to_dict() for c in all_chunks], f, indent=2, ensure_ascii=False)
        
        return all_chunks
    
    def load_processed_chunks(self, doc_id: str) -> List[LegalChunk]:
        """Load previously processed chunks"""
        chunk_file = self.processed_dir / f"{doc_id}_chunks.json"
        if not chunk_file.exists():
            return []
        
        with open(chunk_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [LegalChunk(**item) for item in data]
    
    def get_statistics(self) -> Dict:
        """Get processing statistics - FIXED VERSION"""
        all_chunks = []
        for chunk_file in self.processed_dir.glob("*_chunks.json"):
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:  # ✅ FIXED: Added encoding
                    all_chunks.extend(json.load(f))
            except UnicodeDecodeError as e:
                print(f"⚠️  Warning: Could not read {chunk_file.name}: {e}")
                # Try alternative encoding
                try:
                    with open(chunk_file, 'r', encoding='latin-1') as f:
                        all_chunks.extend(json.load(f))
                        print(f"✓ Successfully read {chunk_file.name} with latin-1 encoding")
                except Exception as e2:
                    print(f"❌ Failed to read {chunk_file.name}: {e2}")
                    continue
        
        if not all_chunks:
            return {
                'total_chunks': 0,
                'total_docs': 0,
                'avg_chunk_length': 0,
                'by_doc_type': {}
            }
        
        stats = {
            'total_chunks': len(all_chunks),
            'total_docs': len(list(self.processed_dir.glob("*_chunks.json"))),
            'avg_chunk_length': sum(len(c['text']) for c in all_chunks) / len(all_chunks) if all_chunks else 0,
            'by_doc_type': {}
        }
        
        for chunk in all_chunks:
            doc_type = chunk['doc_type']
            stats['by_doc_type'][doc_type] = stats['by_doc_type'].get(doc_type, 0) + 1
        
        return stats

# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Process Income Tax Act
    chunks = processor.process_document(
        doc_path="data/raw/income_tax_act_1961.pdf",
        doc_id="it_act_1961",
        doc_type="act",
        doc_title="Income Tax Act, 1961",
        year=2024
    )
    
    # Print statistics
    stats = processor.get_statistics()
    print("\n📊 Processing Statistics:")
    print(f"Total Documents: {stats['total_docs']}")
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Average Chunk Length: {stats['avg_chunk_length']:.0f} characters")
    print(f"By Document Type: {stats['by_doc_type']}")
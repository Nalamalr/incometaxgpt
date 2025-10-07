"""
Legal Document Collector for IncomeTaxGPT
Handles downloading and organizing legal documents
"""

import os
import requests
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime

class LegalDocumentCollector:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.data_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load or create metadata file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"documents": [], "last_updated": None}
    
    def _save_metadata(self):
        """Save metadata to file"""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_document(self, 
                     file_path: str, 
                     doc_type: str,
                     title: str,
                     year: int,
                     description: str = ""):
        """
        Add a document to the collection
        
        Args:
            file_path: Path to the document file
            doc_type: Type (act, rule, circular, notification)
            title: Document title
            year: Year of document
            description: Optional description
        """
        doc_info = {
            "id": len(self.metadata["documents"]) + 1,
            "file_path": str(file_path),
            "doc_type": doc_type,
            "title": title,
            "year": year,
            "description": description,
            "added_date": datetime.now().isoformat()
        }
        
        self.metadata["documents"].append(doc_info)
        self._save_metadata()
        print(f"✓ Added: {title}")
    
    def list_documents(self, doc_type: str = None) -> List[Dict]:
        """List all documents or filter by type"""
        docs = self.metadata["documents"]
        if doc_type:
            docs = [d for d in docs if d["doc_type"] == doc_type]
        return docs
    
    def get_statistics(self):
        """Get collection statistics"""
        docs = self.metadata["documents"]
        stats = {
            "total_documents": len(docs),
            "by_type": {},
            "by_year": {}
        }
        
        for doc in docs:
            # Count by type
            doc_type = doc["doc_type"]
            stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + 1
            
            # Count by year
            year = str(doc["year"])
            stats["by_year"][year] = stats["by_year"].get(year, 0) + 1
        
        return stats

# Example usage
if __name__ == "__main__":
    collector = LegalDocumentCollector()
    
    # Example: Add Income Tax Act
    # After you manually download the PDF, add it like this:
    collector.add_document(
        file_path="data/raw/income_tax_act_1961.pdf",
        doc_type="act",
        title="Income Tax Act, 1961",
        year=2024,  # Latest amended version
        description="Complete Income Tax Act with all amendments up to 2024"
    )
    
    # Example: Add a CBDT circular
    collector.add_document(
        file_path="data/raw/cbdt_circular_2024_01.pdf",
        doc_type="circular",
        title="CBDT Circular No. 1/2024",
        year=2024,
        description="Clarifications on Section 80C deductions"
    )
    
    # Print statistics
    stats = collector.get_statistics()
    print("\n📊 Collection Statistics:")
    print(f"Total Documents: {stats['total_documents']}")
    print(f"By Type: {stats['by_type']}")
    print(f"By Year: {stats['by_year']}")
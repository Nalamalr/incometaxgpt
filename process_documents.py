from src.document_processor import DocumentProcessor
import json

# Initialize processor
processor = DocumentProcessor()

print("="*70)
print("STARTING DOCUMENT PROCESSING")
print("="*70)

# Process Income Tax Act
print("\n📄 Processing Income Tax Act...")
chunks = processor.process_document(
    doc_path="data/raw/income_tax_act_1961.pdf",
    doc_id="it_act_1961",
    doc_type="act",
    doc_title="Income Tax Act, 1961",
    year=2024
)

# Get statistics
stats = processor.get_statistics()
print("\n" + "="*70)
print("📊 PROCESSING COMPLETE")
print("="*70)
print(f"Total Documents Processed: {stats['total_docs']}")
print(f"Total Chunks Created: {stats['total_chunks']}")
print(f"Average Chunk Length: {stats['avg_chunk_length']:.0f} characters")
print(f"\nProcessed files saved in: data/processed/")

# Show sample chunk
if chunks:
    print("\n" + "="*70)
    print("SAMPLE CHUNK:")
    print("="*70)
    sample = chunks[0]
    print(f"Section: {sample.section_number}")
    print(f"Title: {sample.section_title}")
    print(f"Text preview: {sample.text[:200]}...")
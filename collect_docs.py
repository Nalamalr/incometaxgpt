from src.document_collector import LegalDocumentCollector

# Initialize collector
collector = LegalDocumentCollector()

# Add your downloaded documents
collector.add_document(
    file_path="data/raw/income_tax_act_1961.pdf",
    doc_type="act",
    title="Income Tax Act, 1961",
    year=2024,
    description="Complete Income Tax Act with amendments"
)

# Add more documents if you have them
# collector.add_document(
#     file_path="data/raw/cbdt_circular_2024_01.pdf",
#     doc_type="circular",
#     title="CBDT Circular 1/2024",
#     year=2024,
#     description="Clarifications"
# )

# Print statistics
stats = collector.get_statistics()
print("\n" + "="*50)
print("📊 Collection Statistics:")
print("="*50)
print(f"Total Documents: {stats['total_documents']}")
print(f"By Type: {stats['by_type']}")
print(f"By Year: {stats['by_year']}")
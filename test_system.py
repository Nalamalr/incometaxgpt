from src.main_integration import IncomeTaxGPTSystem

print("="*70)
print("INITIALIZING COMPLETE SYSTEM")
print("="*70)

# Initialize
system = IncomeTaxGPTSystem()
system.initialize()

print("\n" + "="*70)
print("TESTING COMPLETE PIPELINE")
print("="*70)

# Test query
query = "What is the maximum deduction I can claim under Section 80C?"
print(f"\n🔍 Query: {query}")
print("\nProcessing...")

response = system.query(query)

print("\n" + "="*70)
print("RESPONSE:")
print("="*70)
print(response['answer'])

print("\n" + "-"*70)
print("METADATA:")
print("-"*70)
print(f"Processing Time: {response['metadata']['processing_time']:.2f}s")
print(f"Number of Sources: {response['metadata']['num_sources']}")
print(f"Query Type: {response['metadata']['query_type']}")

if 'retrieved_chunks' in response:
    print("\n" + "-"*70)
    print("SOURCES:")
    print("-"*70)
    for i, chunk in enumerate(response['retrieved_chunks'][:3], 1):
        print(f"{i}. Section {chunk['section_number']}: {chunk['section_title']}")
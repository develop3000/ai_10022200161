# Author: [Kieron Cameron Neequaye Kotey] | Index: [10022200161]


from src.data_processor import load_and_chunk
from src.retriever import build_index, hybrid_retrieve

# 1. Load data
chunks = load_and_chunk('data/Ghana_Election_Result.csv', 'data/2025-Budget-Statement-and-Economic-Policy_v4.pdf')

# 2. Build indexes
# UPDATE: build_index now creates THREE things (FAISS index, BM25 index, embeddings)
index, bm25, embeddings = build_index(chunks)

# 3. Test a search
# UPDATE: We must pass 'bm25' into the hybrid_retrieve function
results, scores = hybrid_retrieve("What is the main economic policy?", index, bm25, chunks)

print("\n--- TEST SEARCH RESULTS ---")
for i, res in enumerate(results):
    print(f"\nResult {i+1} (Score {scores[i]:.2f}):\n{res[:150]}...")

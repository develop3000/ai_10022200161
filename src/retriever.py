# src/retriever.py
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text):
    """Strips punctuation to ensure BM25 keyword matching doesn't fail on quotes/question marks."""
    return re.sub(r'[^\w\s]', '', text).lower()

def build_index(chunks):
    # A. Build FAISS
    embeddings = model.encode(chunks).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # B. Build BM25 (Using stripped text)
    tokenized_chunks = [clean_text(chunk).split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    return index, bm25, embeddings

def _extract_year_from_query(query: str):
    """Pulls a 4-digit year out of the query string, e.g. '2020'."""
    match = re.search(r'\b(19|20)\d{2}\b', query)
    return match.group(0) if match else None

def hybrid_retrieve(query, index, bm25, chunks, k=5, alpha=0.5):
    """Hybrid Search: Strict Override for CSV, pure Hybrid for dense PDF."""
    query_lower = query.lower()
    clean_query = clean_text(query)

    # ------------------------------------------------------------------ #
    # 1. ELECTION OVERRIDE (For CSV Tabular Data)                        #
    # ------------------------------------------------------------------ #
    forced_csv_chunks = []
    election_keywords = ["election", "won", "winner", "votes", "president", "candidate", "party"]

    if any(kw in query_lower for kw in election_keywords):
        year = _extract_year_from_query(query)
        csv_indices = [i for i, c in enumerate(chunks) if "Ghana Election Result" in c]

        if csv_indices:
            if year:
                year_indices = [i for i in csv_indices if f"Year: {year}" in chunks[i]]
            else:
                year_indices = csv_indices

            if year_indices:
                major_party_chunks = [chunks[i] for i in year_indices if any(p in chunks[i] for p in ["NPP", "NDC"])]
                other_year_chunks = [chunks[i] for i in year_indices if not any(p in chunks[i] for p in ["NPP", "NDC"])]

                def top_bm25(candidate_chunks, n):
                    if not candidate_chunks:
                        return []
                    tokenized = [clean_text(c).split() for c in candidate_chunks]
                    mini_bm25 = BM25Okapi(tokenized)
                    scores = mini_bm25.get_scores(clean_query.split())
                    top_idx = np.argsort(scores)[::-1][:n]
                    return [candidate_chunks[i] for i in top_idx]

                forced_csv_chunks = top_bm25(major_party_chunks, 3) + top_bm25(other_year_chunks, 1)
                forced_csv_chunks = forced_csv_chunks[:k]

    # ------------------------------------------------------------------ #
    # 2. Standard Hybrid Search (For PDF Narrative)                      #
    # ------------------------------------------------------------------ #
    query_vec = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vec, k=50)

    tokenized_query = clean_query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    final_scores = np.zeros(len(chunks))

    for i, chunk_idx in enumerate(indices[0]):
        if chunk_idx != -1:
            vec_score = 1 / (1 + distances[0][i])
            final_scores[chunk_idx] += alpha * vec_score

    max_bm25 = np.max(bm25_scores) if np.max(bm25_scores) > 0 else 1
    for i in range(len(chunks)):
        norm_bm25 = bm25_scores[i] / max_bm25
        final_scores[i] += (1 - alpha) * norm_bm25

    sorted_indices = np.argsort(final_scores)[::-1]

    # ------------------------------------------------------------------ #
    # 3. Combine: CSV override + Hybrid                                  #
    # ------------------------------------------------------------------ #
    final_chunks = []
    final_combined_scores = []

    for c in forced_csv_chunks:
        if c not in final_chunks:
            final_chunks.append(c)
            final_combined_scores.append(1.0)

    for i in sorted_indices:
        if len(final_chunks) >= k:
            break
        if chunks[i] not in final_chunks:
            final_chunks.append(chunks[i])
            final_combined_scores.append(final_scores[i])

    return final_chunks, final_combined_scores
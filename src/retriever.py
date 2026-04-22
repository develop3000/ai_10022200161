# src/retriever.py
# Author: [Your Name] | Index: [Your Index Number]

import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

model = SentenceTransformer('all-MiniLM-L6-v2')


def clean_text(text):
    """Strips punctuation to ensure BM25 keyword matching doesn't fail."""
    return re.sub(r'[^\w\s]', '', text).lower()


def build_index(chunks):
    # A. Build FAISS
    embeddings = model.encode(chunks).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # B. Build BM25
    tokenized_chunks = [clean_text(chunk).split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    return index, bm25, embeddings


def _extract_year_from_query(query: str):
    """Pulls a 4-digit year out of the query string."""
    match = re.search(r'\b(19|20)\d{2}\b', query)
    return match.group(0) if match else None


def hybrid_retrieve(query, index, bm25, chunks, k=5, alpha=0.5):
    """
    Hybrid Search with priority ordering:
    1. National summary chunks (for 'who won' questions)
    2. CSV regional chunks (for specific region/candidate queries)
    3. PDF budget chunks (for budget questions)
    """
    query_lower = query.lower()
    clean_query = clean_text(query)

    # ------------------------------------------------------------------ #
    # 1. NATIONAL SUMMARY OVERRIDE                                        #
    #    For "who won" / "winner" queries, always inject the national     #
    #    summary chunk for the requested year FIRST.                      #
    # ------------------------------------------------------------------ #
    forced_chunks = []
    election_keywords = ["election", "won", "winner", "votes", "president",
                         "candidate", "party", "result"]

    if any(kw in query_lower for kw in election_keywords):
        year = _extract_year_from_query(query)

        # Find all national summary chunks
        summary_indices = [
            i for i, c in enumerate(chunks)
            if "NATIONAL RESULTS SUMMARY" in c
        ]

        if summary_indices:
            if year:
                # Pick the summary for the specific year asked about
                year_summary = [
                    chunks[i] for i in summary_indices
                    if f"Ghana {year} Presidential Election" in chunks[i]
                ]
                forced_chunks = year_summary[:1]  # Only need 1 summary per year
            else:
                # No year specified — include most recent summary (2020)
                forced_chunks = [chunks[summary_indices[-1]]]

        # Also add top NPP/NDC regional chunks for that year as supporting evidence
        if year:
            csv_indices = [
                i for i, c in enumerate(chunks)
                if "Ghana Election Result Record" in c
                and "NATIONAL RESULTS SUMMARY" not in c
                and f"Year: {year}" in c
            ]
            major_party = [
                chunks[i] for i in csv_indices
                if any(p in chunks[i] for p in ["NPP", "NDC"])
                and "Code: NPP" in chunks[i] or "Code: NDC" in chunks[i]
            ]

            def top_bm25_local(candidate_chunks, n):
                if not candidate_chunks:
                    return []
                tokenized = [clean_text(c).split() for c in candidate_chunks]
                mini_bm25 = BM25Okapi(tokenized)
                scores = mini_bm25.get_scores(clean_query.split())
                top_idx = np.argsort(scores)[::-1][:n]
                return [candidate_chunks[i] for i in top_idx]

            regional_support = top_bm25_local(major_party, k - len(forced_chunks) - 1)
            forced_chunks = forced_chunks + regional_support

        forced_chunks = forced_chunks[:k]

    # ------------------------------------------------------------------ #
    # 2. Standard Hybrid Search (For PDF and remaining slots)            #
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
    # 3. Combine: forced priority chunks + hybrid remainder              #
    # ------------------------------------------------------------------ #
    final_chunks = []
    final_scores_out = []

    for c in forced_chunks:
        if c not in final_chunks:
            final_chunks.append(c)
            final_scores_out.append(1.0)

    for i in sorted_indices:
        if len(final_chunks) >= k:
            break
        if chunks[i] not in final_chunks:
            final_chunks.append(chunks[i])
            final_scores_out.append(final_scores[i])

    return final_chunks, final_scores_o
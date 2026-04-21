# app.py
import streamlit as st
from src.data_processor import load_and_chunk
from src.retriever import build_index, hybrid_retrieve
from src.generator import generate_response, construct_prompt
from src.logger import log_interaction

# 1. Page Setup
st.set_page_config(page_title="Ghana RAG System", page_icon="🇬🇭", layout="wide")
st.title("🇬🇭 Ghana Election & Budget AI Assistant")

# --- PART G: INNOVATION (System Admin Sidebar) ---
with st.sidebar:
    st.header("📂 System Admin")
    try:
        with open("logs/experiment_logs.txt", "r", encoding="utf-8") as f:
            log_data = f.read()
        st.download_button("Download Audit Logs", log_data, "rag_logs.txt")
    except:
        st.caption("No logs yet.")

# Explicitly define the file paths so NameError is impossible
CSV_FILE_PATH = 'data/Ghana_Election_Result.csv'
PDF_FILE_PATH = 'data/2025-Budget-Statement-and-Economic-Policy_v4.pdf'

# 2. Setup System
@st.cache_resource
def initialize_system():
    """
    This function only runs ONCE per session (or when files change).
    The @st.cache_resource decorator ensures it doesn't re-run on every button click.
    """
    chunks = load_and_chunk(CSV_FILE_PATH, PDF_FILE_PATH)
    index, bm25, _ = build_index(chunks)
    return chunks, index, bm25

# Load the chunks into memory
# The spinner is OUTSIDE the cached function so it only shows during actual builds
if 'system_loaded' not in st.session_state:
    with st.spinner("🔄 Building Knowledge Base (First Time Only)..."):
        chunks, index, bm25 = initialize_system()
    st.session_state.system_loaded = True
    st.success("✅ System Ready! Knowledge base is now in memory.")
else:
    # Silent cache retrieval - no spinner needed
    chunks, index, bm25 = initialize_system()

# --- 🚨 HOLISTIC SYSTEM AUDIT ---
with st.expander("🚨 SYSTEM AUDIT: WHAT DOES THE APP ACTUALLY KNOW?"):
    st.write(f"Total Knowledge Chunks in Memory: {len(chunks)}")
    
    
    pdf_chunks = [c for c in chunks if "Ghana 2025 Budget Document" in c]
    st.write(f"Budget PDF Chunks: {len(pdf_chunks)}")
    
    if len(pdf_chunks) > 0:
     st.success("✅ Budget PDF loaded! Sample:")
     st.caption(pdf_chunks[0][:300])
    else:
     st.error("❌ CRITICAL: 0 budget chunks in memory.")
    
    csv_chunks = [c for c in chunks if "Ghana Election Result" in c]
    st.write(f"Election Records Successfully Loaded: {len(csv_chunks)}")
    
    if len(csv_chunks) > 0:
        st.success("CSV is loaded! Here are the first two records it sees:")
        st.write(csv_chunks[:2])
    else:
        st.error("CRITICAL FAILURE: The system has 0 election records in memory.")
# -----------------------------------------------------

# 3. User Input
query = st.text_input("Ask a question about the 2025 Budget or Election Results:")

if st.button("Search"):
    if query:
        with st.spinner("🔍 Consulting documents..."):
            # A. Retrieve
            retrieved_chunks, scores = hybrid_retrieve(query, index, bm25, chunks)
            
            # B. Generate (Real AI Response)
            response = generate_response(query, retrieved_chunks)
            
            # C. Log
            log_interaction(query, retrieved_chunks, response)
            
            # D. Display Result
            st.subheader("AI Response")
            st.info(response)
            
            # E. Diagnostics
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("View Retrieved Context"):
                    for i, chunk in enumerate(retrieved_chunks):
                        st.write(f"**Source {i+1} (Score: {scores[i]:.2f})**")
                        st.caption(chunk)
            with col2:
                with st.expander("View Final Prompt"):
                    st.code(construct_prompt(query, retrieved_chunks))
    else:
        st.warning("Please enter a question.")
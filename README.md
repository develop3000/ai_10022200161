# 🇬🇭 Ghana Election & Budget RAG System

**Course:** CS4241 — Introduction to Artificial Intelligence  
**Institution:** Academic City University  
**Lecturer:** Godwin N. Danso  

###  Student Details
* **Name:** Kieron Cameron Neequaye Kotey
* **Index Number:** 10022200161

---

###  Final Deliverables Links
* **Video Walkthrough (Max 2 mins):** [INSERT YOUTUBE/DRIVE LINK HERE]
* **Deployed Cloud Application:** [INSERT DEPLOYED APP URL HERE]
* **Full Project Documentation:** See `Artificial Intelligence Exams-10022200161.pdf` in the root directory.

---

##  Project Overview

This repository contains a fully custom **Retrieval-Augmented Generation (RAG)** chatbot designed to answer complex queries regarding the 2025 Ghana Budget Statement (unstructured PDF) and historical Ghana Election Results (structured CSV).

** CRITICAL CONSTRAINT MET:** This system was built **100% from scratch**. No end-to-end frameworks like LangChain, LlamaIndex, or pre-built RAG pipelines were used. All core RAG components—including document cleaning, sliding-window chunking, vector embeddings, hybrid retrieval math, and prompt construction—were implemented manually using foundational libraries.

---

## Part G: Innovation Component (The Two-Brain Router)
Standard unified RAG architectures suffer from "context starvation" when sparse tabular data (CSV) competes mathematically against dense narrative text (PDF). 

To solve this, this project features a novel **Domain-Specific Scoring Function** (The Strict Two-Brain Router). The system utilizes an active intent-classification layer that preemptively intercepts queries:
* **Domain 1 (Election Override):** Dynamically blocks PDF context and executes a specialized BM25 search on mathematically aggregated "National Summary" chunks.
* **Domain 2 (Semantic Budget Filter):** Blocks CSV data and strictly prioritizes deep narrative semantics by hardcoding the Hybrid Search weight to `Alpha = 0.8`.

---

## Project Structure

```text
ai_10022200161/
│
├── app.py                     # Streamlit frontend UI
├── evaluate_rag.py            # Automated adversarial testing suite
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (Add your Hugging Face Token here)
│
├── data/                      # Knowledge Base Documents
│   ├── Ghana_Election_Result.csv
│   └── 2025-Budget-Statement-and-Economic-Policy_v4.pdf
│
├── src/                       # Core RAG Modules (Built from scratch)
│   ├── data_processor.py      # Custom chunking, PDF cleaning & National Aggregation
│   ├── retriever.py           # FAISS Vector DB, BM25, and Hybrid 'Two-Brain' Router
│   ├── generator.py           # Prompt construction and Hugging Face API integration
│   └── logger.py              # System interaction logging
│
└── logs/                      # Audit trails
    ├── experiment_logs.txt    # Human-in-the-loop manual testing and debug logs
    └── evaluation_report.txt  # Automated RAG vs. Pure LLM adversarial metrics

# src/data_processor.py
import pandas as pd
import fitz
import os
import re

def clean_pdf_text(text):
    """
    Cleans raw PDF text extracted from a budget document.
    Removes noise while preserving meaningful content.
    """
    # Remove lines that are ONLY numbers, spaces, dashes, or currency symbols
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip lines that are purely numeric/table noise
        # A "meaningful" line must have at least 3 consecutive alphabetic characters
        if not re.search(r'[a-zA-Z]{3,}', line):
            continue
        # Skip lines that are mostly numbers (>60% digits/punctuation)
        alpha_chars = sum(1 for c in line if c.isalpha())
        total_chars = len(line)
        if total_chars > 0 and (alpha_chars / total_chars) < 0.25:
            continue
        # Skip very short lines (page numbers, section markers)
        if len(line) < 20:
            continue
        cleaned_lines.append(line)
    
    return ' '.join(cleaned_lines)


def load_and_chunk(csv_path, pdf_path):
    """
    Processes CSV and PDF data with a proper cleaning pipeline.
    """
    chunks = []

    # 1. CSV INGESTION (unchanged - works fine)
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                row_data = ", ".join([f"{col}: {val}" for col, val in row.items()])
                chunks.append(f"Ghana Election Result Record: {row_data}")
            print(f"✅ CSV: {len(df)} records loaded.")
        except Exception as e:
            print(f"❌ CSV Processing Error: {e}")
    else:
        print(f"❌ CSV not found: {os.path.abspath(csv_path)}")

    # 2. PDF INGESTION with cleaning pipeline
    print(f"📄 PDF path: {os.path.abspath(pdf_path)}")
    if not os.path.exists(pdf_path):
        print(f"❌ PDF NOT FOUND. Check your 'data' folder.")
        return chunks

    try:
        doc = fitz.open(pdf_path)
        print(f"✅ PDF opened: {len(doc)} pages")

        pdf_count = 0
        raw_chars = 0
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            raw_text = page.get_text("text")
            raw_chars += len(raw_text)
            
            # Run the cleaning pipeline
            cleaned_text = clean_pdf_text(raw_text)
            
            if not cleaned_text or len(cleaned_text) < 60:
                continue  # Skip empty or near-empty pages (cover pages, blank pages)

            # Sliding window chunking on the cleaned text
            size = 700
            overlap = 150
            for i in range(0, len(cleaned_text), size - overlap):
                chunk = cleaned_text[i:i + size].strip()
                if len(chunk) > 80:
                    # Label each chunk so the retriever knows it's from the budget PDF
                    chunks.append(f"Ghana 2025 Budget Document: {chunk}")
                    pdf_count += 1

        doc.close()
        print(f"✅ Raw PDF characters extracted: {raw_chars}")
        print(f"✅ Clean PDF chunks created: {pdf_count}")

        if pdf_count == 0:
            print("⚠️  WARNING: 0 PDF chunks created. The PDF may still be corrupted.")

    except Exception as e:
        print(f"❌ PDF Processing Error: {e}")

    return chunks
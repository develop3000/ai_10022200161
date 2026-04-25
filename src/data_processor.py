# src/data_processor.py
# Author: [Kieron Cameron Neequaye Kotey] | Index: [10022200161]


import pandas as pd
import fitz
import os
import re


def clean_pdf_text(text):
    """
    Cleans raw PDF text extracted from a budget document.
    Removes noise while preserving meaningful content.
    """
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not re.search(r'[a-zA-Z]{3,}', line):
            continue
        alpha_chars = sum(1 for c in line if c.isalpha())
        total_chars = len(line)
        if total_chars > 0 and (alpha_chars / total_chars) < 0.25:
            continue
        if len(line) < 20:
            continue
        cleaned_lines.append(line)

    return ' '.join(cleaned_lines)


def build_national_summary_chunks(df: pd.DataFrame) -> list:
    """
    NEW FUNCTION — Fixes the core accuracy problem.

    The CSV only has regional breakdowns. When a user asks "who won the 2020
    election?", the retriever returns 5 random regional rows, and the LLM
    can only summarise those regions — not the national result.

    This function aggregates all regional votes into NATIONAL totals for
    each election year and candidate, then creates a single rich summary
    chunk per year. These summary chunks are inserted into the knowledge
    base alongside the raw regional rows, giving the LLM the full picture.
    """
    summary_chunks = []
    years = sorted(df['Year'].unique())

    for year in years:
        year_df = df[df['Year'] == year]

        # Aggregate all regional votes into national totals per candidate
        national = (
            year_df.groupby(['Candidate', 'Party', 'Code'])['Votes']
            .sum()
            .reset_index()
        )
        total_votes = national['Votes'].sum()
        national['NationalPct'] = (national['Votes'] / total_votes * 100).round(2)
        national = national.sort_values('Votes', ascending=False)

        # Build a readable summary string
        lines = [f"Ghana {year} Presidential Election — NATIONAL RESULTS SUMMARY:"]
        lines.append(f"Total votes cast nationally: {total_votes:,}")
        lines.append("National results by candidate (all regions combined):")

        winner = None
        for _, row in national.iterrows():
            candidate = row['Candidate']
            party = row['Party']
            votes = int(row['Votes'])
            pct = row['NationalPct']
            lines.append(
                f"  {candidate} ({party}): {votes:,} votes — {pct}% of national vote"
            )
            if winner is None and row['Code'] in ['NPP', 'NDC']:
                winner = (candidate, party, pct)

        # Explicitly state the winner for easy LLM retrieval
        if winner:
            lines.append(
                f"WINNER: Based on national aggregate, {winner[0]} of the "
                f"{winner[1]} party led with {winner[2]}% of the total national vote."
            )

        chunk_text = " | ".join(lines)
        summary_chunks.append(
            f"Ghana Election Result Record: {chunk_text}"
        )

    return summary_chunks


def load_and_chunk(csv_path, pdf_path):
    """
    Processes CSV and PDF data with a proper cleaning pipeline.
    Now includes national aggregation summaries for accurate election answers.
    """
    chunks = []

    # ── 1. CSV INGESTION ─────────────────────────────────────────────────────
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)

            # A. Add NATIONAL SUMMARY chunks first (highest priority)
            summary_chunks = build_national_summary_chunks(df)
            chunks.extend(summary_chunks)
            print(f"✅ CSV: {len(summary_chunks)} national summary chunks created "
                  f"(one per election year).")

            # B. Add individual regional row chunks (for detailed regional queries)
            regional_count = 0
            for _, row in df.iterrows():
                row_data = ", ".join([f"{col}: {val}" for col, val in row.items()])
                chunks.append(f"Ghana Election Result Record: {row_data}")
                regional_count += 1

            print(f"✅ CSV: {regional_count} regional records loaded.")

        except Exception as e:
            print(f"❌ CSV Processing Error: {e}")
    else:
        print(f"❌ CSV not found: {os.path.abspath(csv_path)}")

    # ── 2. PDF INGESTION ─────────────────────────────────────────────────────
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

            cleaned_text = clean_pdf_text(raw_text)

            if not cleaned_text or len(cleaned_text) < 60:
                continue

            size = 700
            overlap = 150
            for i in range(0, len(cleaned_text), size - overlap):
                chunk = cleaned_text[i:i + size].strip()
                if len(chunk) > 80:
                    chunks.append(f"Ghana 2025 Budget Document: {chunk}")
                    pdf_count += 1

        doc.close()
        print(f"✅ Raw PDF characters extracted: {raw_chars}")
        print(f"✅ Clean PDF chunks created: {pdf_count}")

        if pdf_count == 0:
            print("⚠️  WARNING: 0 PDF chunks created.")

    except Exception as e:
        print(f"❌ PDF Processing Error: {e}")

    return chunks

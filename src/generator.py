# src/generator.py
# Author: [Kieron Cameron Neequaye Kotey] | Index: [10022200161]

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def construct_prompt(query, retrieved_chunks):
    """
    Constructs a prompt that correctly handles both:
    - National election winner questions (uses NATIONAL RESULTS SUMMARY chunks)
    - Regional/specific election questions (uses regional CSV rows)
    - Budget policy questions (uses PDF chunks)
    """
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are an expert AI assistant specialising in Ghana Election results and Ghana Budget policy analysis.
Use ONLY the following retrieved context to answer the user's question.

CRITICAL RULES:
1. ELECTION WINNER QUESTIONS: If the context contains a chunk labelled "NATIONAL RESULTS SUMMARY", 
   use it to give the OVERALL national winner with their total national vote percentage. 
   Do NOT say "only in Western Region" or limit your answer to one region — use the national totals.

2. REGIONAL QUESTIONS: If the user asks about a specific region, answer using the regional records.

3. BUDGET QUESTIONS: Use the budget document chunks to answer policy questions accurately.

4. INSUFFICIENT INFORMATION: If the answer is genuinely not in the context, say exactly: 
   "I do not have sufficient information."

5. NO FABRICATION: Never invent facts, figures, or candidates not present in the context.

6. BE CONCISE: Give a direct, professional answer. For election winners, state the candidate, 
   party, and their national vote percentage clearly.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
    return prompt


def generate_response(query, retrieved_chunks):
    """Calls Hugging Face API with an automatic fallback loop for stability."""
    prompt = construct_prompt(query, retrieved_chunks)

    if not HF_TOKEN:
        return "⚠️ Error: HF_TOKEN missing in .env file."

    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "mistralai/Mistral-Nemo-Instruct-2407"
    ]

    last_error = ""

    for model_id in models:
        try:
            client = InferenceClient(model=model_id, token=HF_TOKEN)
            messages = [{"role": "user", "content": prompt}]

            response = client.chat_completion(
                messages,
                max_tokens=500,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            last_error = str(e)
            continue

    return (f"❌ All free AI servers are currently busy. Please try again in 30 seconds.\n"
            f"(Last Error: {last_error})")

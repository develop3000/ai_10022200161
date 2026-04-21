# src/generator.py
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def construct_prompt(query, retrieved_chunks):
    """Constructs a prompt that encourages summarizing regional data."""
    context = "\n\n".join(retrieved_chunks)
    
    prompt = f"""You are an expert AI assistant for Ghana Election results and Budget policy analysis. 
Use ONLY the following retrieved context to answer the user's question. 

CRITICAL RULES:
1. If the context contains regional or constituency results for the requested year, summarize who the leading candidate is based on those records.
2. If the answer is absolutely not in the context, strictly say "I do not have sufficient information."
3. Do not make up facts. Keep answers concise and professional.

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

    # A robust list of the most active "Chat" models currently on Hugging Face.
    # If a server drops one, the system automatically catches the error and tries the next.
    models = [
        "Qwen/Qwen2.5-7B-Instruct",             # Top tier, highly available right now
        "meta-llama/Llama-3.2-3B-Instruct",     # Excellent, fast fallback
        "microsoft/Phi-3-mini-4k-instruct",     # Very lightweight, rarely goes down
        "mistralai/Mistral-Nemo-Instruct-2407"  # Standard reliable backup
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
            # If the model fails (like your Zephyr error), save the error and try the next model
            last_error = str(e)
            continue
            
    # If we loop through all 4 and they are ALL down, show this message
    return f"❌ All free AI servers are currently busy. Please try again in 30 seconds.\n(Last Error: {last_error})"
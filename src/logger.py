# src/logger.py
# Author: [Kieron Cameron Neequaye Kotey] | Index: [10022200161]

import datetime
import os

# Create a logs folder if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

def log_interaction(query, retrieved_chunks, response):
    """
    Saves the user's question, the retrieved context, and the AI's answer to a file.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = f"""
=========================================
TIME: {timestamp}
USER QUERY: {query}
-----------------------------------------
RETRIEVED CONTEXT (Top Chunks):
{retrieved_chunks}
-----------------------------------------
FINAL RESPONSE:
{response}
=========================================
\n"""
    
    # Append the log entry to the history file
    with open("logs/experiment_logs.txt", "a", encoding="utf-8") as file:
        file.write(log_entry)

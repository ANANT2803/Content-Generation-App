
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CHANGED: Import Google Class ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from prompts import final_prompt

# --- 1. Setup Google Client ---
# LangChain automatically looks for "GOOGLE_API_KEY" in your environment, 
# so we don't strictly need to fetch it manually, but it's good practice to check.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# --- 2. Helper Functions (Unchanged) ---
def create_history(messages):
    """
    Formats the chat history from Streamlit into a format LangChain understands.
    """
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history

# --- 3. Chain Construction ---

def get_chain():
    """
    Creates the Content Generation Chain using Gemini.
    """
    
    # --- CHANGED: Initialize Gemini ---
    # We use 'gemini-1.5-pro' for high-quality copywriting.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        max_retries=2
        # api_key=GOOGLE_API_KEY # Optional if env var is set
    )

    # The Chain:
    # 1. Takes inputs
    # 2. Formats using 'final_prompt'
    # 3. Sends to Gemini
    # 4. Parses output
    chain = (
        final_prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain

# --- 4. Main Invocation Function (Unchanged) ---
def invoke_chain(user_input, messages):
    chain = get_chain()
    history = create_history(messages)

    response = chain.invoke(
        {
            "input": user_input,
            "messages": history.messages
        }
    )
    return response

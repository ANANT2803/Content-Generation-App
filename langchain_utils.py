
# Content Generation RAG
import os
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from prompts import final_prompt

# --- 1. Setup Azure Client ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_API_BASE") 
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

# --- 2. Helper Functions ---

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
    Creates the Content Generation Chain.
    Structure: User Input + History -> Creative Prompt -> Azure LLM -> String Output
    """
    
    # Initialize the LLM
    # We set temperature=0.7 to allow for creativity
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL_NAME,
        temperature=0.7 
    )

    # The Chain:
    # 1. Takes the inputs (question, history)
    # 2. Formats them using 'final_prompt' (which includes the smart example selector)
    # 3. Sends it to the Azure LLM
    # 4. Parses the output as a simple string
    chain = (
        final_prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain

# --- 4. Main Invocation Function ---

def invoke_chain(user_input, messages):
    """
    The main function called by Streamlit.
    """
    chain = get_chain()
    history = create_history(messages)

    # We map the user's input to the key "input" because that is what 
    # we defined in your prompts.py file ({input})
    response = chain.invoke(
        {
            "input": user_input,
            "messages": history.messages
        }
    )
    return response

# Prompt->LLM->Parse/Clean
# examples->prompt->langchain
# prompts >> fine-tuning

# main.py (streamlit)

# Cont Gen..

import streamlit as st
from langchain_utils import invoke_chain

# import only if sidebar me u want to see which example is being retrieved and passed to LLM
from examples import get_example_selector

# 1. Update the Title

# st.set_page_config(page_title="AI Content Agent", page_icon="✍️")
st.title("SAIC Content Generator")
# st.caption("Generate posts for LinkedIn, Instagram, or Emails using your custom style.")
# u can find and implement more such fxns to further enhance the UI, documentation dekhlena streamlit

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Update Input Placeholder (:= this symbol just assigns the users input to this prompt variable we defined)
if prompt := st.chat_input("Draft a LinkedIn post about... "):

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # # (Uncomment agr debug krke dekhna ho wrna chill hai ye part) This runs the Semantic Search manually just to show you which example it found and is passed to the LLM
    # with st.sidebar:
    #     st.header("🔍 Semantic Search Debugger")
    #     st.write(f"**Current Input:** {prompt}")
    #     st.divider()
        
    #     # 1. Get the selector tool
    #     selector = get_example_selector()
        
    #     # 2. Ask it: "Which examples match this input?"
    #     selected_examples = selector.select_examples({"input": prompt})
        
    #     # 3. Display the results
    #     st.write(f"**Found {len(selected_examples)} Relevant Example(s):**")
        
    #     for i, example in enumerate(selected_examples):
    #         with st.expander(f"Example #{i+1}", expanded=True):
    #             st.caption(f"Similiarity Match #{i+1}")
    #             st.markdown(f"**Input:**\n> {example['input']}")
    #             st.markdown(f"**Response Style:**\n> {example['response'][:100]}...") # (Shows just the first 100 chars to keep it clean, increase krlena length if u want)
    # # ------------------------- 

    # 3. Update Spinner Text
    with st.spinner("Crafting your content..."):
        with st.chat_message("assistant"):
            # Call the backend
            response = invoke_chain(prompt, st.session_state.messages)
            st.markdown(response)

    # Adding assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

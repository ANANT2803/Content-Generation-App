# prompt.py
from examples import get_example_selector
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate
)

def get_content_prompt():
    """
    Returns a LangChain ChatPromptTemplate ready for generating
    alumni posts, announcements, and event content.
    """

    # Get the example selector from examples.py
    example_selector = get_example_selector()

    # Few-shot example template: dynamically fetches the most relevant example
    few_shot_examples = FewShotChatMessagePromptTemplate(
        example_selector=example_selector,
        example_prompt=ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{query}")
        ]),
        input_variables=["input"]
    )

    # Final prompt with system instructions + dynamic examples
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """
         You are an expert content generation.
          

         Guidelines for content generation:
         - Tone: professional yet warm, engaging, and human-like
         - Style: concise, celebratory, inspiring
         - Include: names, achievements, dates, links, and key details
         - Optional: emojis, hashtags, and subtle formatting for social media
         
         Match the tone and style of the examples provided.
         """),
        MessagesPlaceholder(variable_name="examples"),  # Insert relevant few-shot examples
        ("human", "{input}")  # User input goes here
    ])

    return final_prompt, few_shot_examples

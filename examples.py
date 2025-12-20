import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import AzureOpenAIEmbeddings

examples = [
    {
        "input": "Generate an Instagram content for the alumni achievement post for our alumnus, 'Mr. Pratyush Choudhury (BME '21)' who co-founded 'Activate VC' and raised '75 million USD' in funding"
        "query": "SAIC is proud to share that Mr. Pratyush Choudhury (BME ‘21), alumnus of IIT (BHU), Varanasi, has co-founded Activate VC, India’s first AI-focused venture fund, with an impressive Fund I of $75 million, leveraging his exceptional trajectory in the venture ecosystem to become one of the youngest General Partners in the country, aiming to empower India’s next generation of AI founders, and we extend our heartfelt congratulations to him on this remarkable achievement."

    },
   {
       "input": "Generate an instagram content for the alumni achievement post for our alumnus, 'Mr.Prateek Maheshwari' for his role in the 'Physics Wallah' IPO."   
       "query": "SAIC is proud to celebrate this incredible achievement as we congratulate Prateek Maheshwari for being an important part of Physics Wallah’s spectacular IPO, highlighting his journey from campus corridors to contributing to one of India’s most inspiring ed-tech journeys, and here’s to innovation, impact, and many more milestones ahead."
   },
   {
       
      "input": "Generate a warm, invitation-style announcement post for 'Connect & Learn' happening in the 'Pink City' (Jaipur) this summer. The post should highlight the goal of providing clarity and connection for students navigating new challenges and emphasize that experiences will be shared and ideas challenged. Include all specified event details (Date: 11 July, Friday,Time: 6:00 PM Onwards)",
      "query": "Still figuring things out? You’re not alone. New cities, new skills, new questions—it’s all part of the ride. That’s where Connect & Learn steps in. This summer, SAIC is bringing students and alumni together in the Pink City—where centuries of history echo through bold, new voices and stories flow as easily as sand. It’s where experiences are unpacked, ideas are challenged, and every interaction has the potential to shape what comes next. If you are seeking connection, clarity, a sense of direction or just the comfort of being understood, this is the place for you. Event Details: Date: 11 July, Friday Time: 6:00 PM Onwards Venue: https://maps.app.goo.gl/Zei5JHDz9ox8Cw989?g_st=ipc\n\nSeats are limited. Register now through the link in our bio."

   },
   {
       
     "input": "Generate an alumni achievement post for alumni Dirghayu Kaushik and Vikrant Shivalik, founders of Ambitio (AI study abroad platform). Mention their $2M funding and mission to transform global education."
     "query": "Built on friendship, fueled by resilience, What started as a dream is now shaping the future of global education! SAIC is proud to share the success of Mr. Dirghayu Kaushik (PHE ‘21), Founder & CEO, and Mr. Vikrant Shivalik (PHE ‘21), Founder & COO. They are the visionaries behind Ambitio, an AI-powered platform transforming the study abroad experience, making global education more accessible and seamless. Ambitio has secured $2 million in funding to drive this revolution. We wish them continued success in transforming the future of study abroad! "

   },
   {
    
    "input": "Generate an alumni achievement post from SAIC for alumnus Mr. Siddharth Jain (IIT BHU) on his appointment as Managing Partner and Country Head of Kearney. Highlight his 17 years of consulting experience, focus on operational excellence/long-term success, and his impact across India, SEA, ME, and Africa.",
    "query": "SAIC is elated to share that Mr. Siddharth Jain, an alumnus of IIT (BHU), Varanasi has been appointed as Managing Partner and Country Head of Kearney, a leading global management consulting firm. Mr. Jain provides over 17 years of excellent consulting experience, constantly excelling at increasing operational excellence and creating long-term success. His strategic acumen and significant achievements have echoed well beyond India, reaching Southeast Asia, the Middle East, and Africa. His incredible path demonstrates the value of endurance, vision, and leadership, serving as a beacon for our students and alumni. We extend our warmest congratulations to him on this remarkable milestone and wish him continued success in driving meaningful impact across industries and communities."

   },
   {
       
    "input": "Generate a congratulatory post from SAIC for alumna Ms. Gayatri Bandi (MIN ’22) for being named to the WIM UK 2024 '100 Global Inspirational Women In Mining' list. Highlight her distinction as the first female underground mining engineer in the 117-year history of the Jharia coalfields. Mention her advocacy for inclusivity as the head of the Women of MEAI.",
    "query": "SAIC proudly announces that Ms. Gayatri Bandi (MIN ’22), a pioneer in mining and inclusivity, has been recognized on the Women In Mining UK (WIM UK) 2024 list of ‘100 Global Inspirational Women In Mining.’ She serves as the Manager in Tata Steel’s Mining Division and is the first female underground mining engineer in the 117-year history of the Jharia coalfields. A recipient of the Roberton Medal, she has led vital projects in a male-dominated field. Additionally, Ms. Gayatri passionately advocates for inclusivity as the head of the Women of MEAI (Mining Engineers Association of India). This prestigious recognition highlights the exceptional contributions of women in the mining industry."

   }
   

]

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_BASE = os.getenv("AZURE_API_BASE")
AZURE_API_VERSION = os.getenv("AZURE_OPEN_API_VERSION")
AZURE_MODEL_NAME = os.getenv("AZURE_MODEL_NAME")

# Embeddings Config
AZURE_EMBEDDINGS_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
AZURE_API_VERSION_EMBEDDING = os.getenv("AZURE_API_VERSION_EMBEDDING")

@st.cache_resource
def get_example_selector():
    """
    Creates a smart selector that finds the most relevant examples 
    from the list above based on the semantic meaning of the user's input.
    """
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        AzureOpenAIEmbeddings(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_API_BASE,
            api_version=AZURE_API_VERSION_EMBEDDING,
            azure_deployment=AZURE_EMBEDDINGS_MODEL_NAME,
        ),
        Chroma, # Vector Store
        k=1, # We only need 1 or 2 good examples for style matching
        input_keys=["input"],
    )
    return example_selector

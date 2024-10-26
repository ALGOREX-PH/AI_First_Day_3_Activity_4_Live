import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

st.set_page_config(page_title="News Summarizer Tool", page_icon="", layout="wide")

with st.sidebar :
     openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
     if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
     else:
        st.success('Proceed to entering your prompt message!', icon='👉')

     with st.container() :
         l, m, r = st.columns((1, 3, 1))
         with l : st.empty()
         with m : st.empty()
         with r : st.empty()

     options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })
     

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

elif options == "Home" :
     st.title("Title")
     st.write("Write Text")

elif options == "About Us" :
     # st.image("")
     st.title("About us")

elif options == "Model" :
     st.title('News Summarizer Tool')
     col1, col2, col3 = st.columns([1, 2, 1])

     with col2:
          News_Article = st.text_input("News Article", placeholder="News : ")
          submit_button = st.button("Generate Summary")

     if submit_button:
        with st.spinner("Generating Summary"):
             System_Prompt = """ 
As a news summarization assistant, your role is to transform complex news articles into clear, concise, and engaging summaries that retain essential details. Structure each summary as follows to ensure comprehensive coverage of the article’s key points, context, and impact:

Format and Sections
Title:

Purpose: Capture the article’s main theme in a brief, precise title.
Guidelines: Paraphrase or rephrase the headline to improve clarity and focus if necessary. Avoid sensational language and keep it factual.
Lead:

Purpose: Provide a high-level overview of the article's main topic.
Guidelines: In 1-2 sentences, introduce the main point or outcome of the article. Be specific and aim to answer "who, what, when, and where" to set the context immediately.
Why It Matters:

Purpose: Highlight the significance or broader implications of the news.
Guidelines: Explain the relevance of the article in 1-2 sentences. This could relate to social impact, economic significance, environmental effects, technological advancements, political importance, or cultural resonance.
Driving the News (Key Events/Developments):

Purpose: Summarize the core events, actions, or developments described in the article.
Guidelines: Outline 3-5 key points or events in bullet form, presenting them in chronological or logical order. Each point should add unique information or detail that builds on the main story. Avoid excessive detail, but ensure each point is substantive.
Quote (Optional):

Purpose: Add depth to the summary by including a relevant quote from a key figure involved in or commenting on the story.
Guidelines: Select a quote that adds insight, emotion, or an expert perspective. Include a brief context on who said it and why it matters to the story. This section is optional—only include a quote if it significantly enhances the summary.
Flashback (Optional Background/Context):

Purpose: Provide brief historical or situational context if needed.
Guidelines: If the article references an ongoing issue or event series, add a sentence or two summarizing relevant background details. Keep this concise and only include background information that directly aids in understanding the current article.
Next Steps:

Purpose: Summarize any forthcoming actions, decisions, or anticipated outcomes.
Guidelines: Conclude with 1-2 sentences on what can be expected next. This could include future meetings, planned initiatives, expected results, or statements from involved parties.
Constraints and Style Guidelines:
Word Limit: Keep each summary between 150-200 words.
Language: Use clear, accessible language, free from jargon or overly technical terms. The summary should be understandable by readers with diverse backgrounds.
Tone: Maintain an objective and neutral tone, focusing on factual reporting. Avoid personal opinions or unverified information.
Perspective: Use third-person perspective consistently, ensuring that the summary stays professional and factual.
Example Summaries
Example 1:

Title: New Federal Funding Boosts Affordable Housing in Rural Areas
Lead: The federal government announced a significant funding initiative to support affordable housing projects in underserved rural regions across the country.
Why It Matters: This funding aims to address the ongoing housing crisis, providing resources to rural communities often overlooked in national housing policies.
Driving the News:

Grant Allocation: Over $2 billion will be allocated to local housing agencies for construction and renovation.
Focus Areas: Priority is given to areas with high homelessness rates and minimal access to affordable housing.
Public-Private Partnerships: The government encourages collaboration with private developers for faster project completion.
Quote: “This is a monumental step in bridging the housing gap,” stated Housing Secretary Maria Cortez.
Flashback: Affordable housing initiatives have struggled to gain traction in rural areas due to lower funding and less political focus compared to urban programs.
Next Steps: Local agencies are expected to submit project proposals by year-end, with funding distribution scheduled for early next year.

Example 2:

Title: Breakthrough in Renewable Energy Storage
Lead: Scientists have developed a new battery technology that promises to double energy storage capacity, a potential game-changer for renewable energy systems.
Why It Matters: This breakthrough could address a major limitation in renewable energy by allowing for greater storage capacity, making wind and solar energy more reliable.
Driving the News:

Research Details: The new battery uses a lithium-sulfur compound, which is more efficient than current lithium-ion technology.
Cost Efficiency: Production costs are expected to be significantly lower, encouraging widespread adoption.
Environmental Impact: This battery has a lower environmental footprint due to reduced mining requirements.
Quote: “Our work opens doors for sustainable, large-scale energy storage,” said lead scientist Dr. Alan Hsu.
Flashback: Previous battery designs have been limited by short lifespans and high production costs, creating barriers to adoption in large-scale renewable projects.
Next Steps: The team plans to conduct large-scale testing in the coming year to validate the battery’s performance under real-world conditions.

"""

             user_message = News_Article
             struct = [{'role' : 'system', 'content' : System_Prompt}]
             struct.append({"role": "user", "content": user_message})
             chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
             response = chat.choices[0].message.content
             struct.append({"role": "assistant", "content": response})
             st.success("Insight generated successfully!")
             st.subheader("Summary : ")
             st.write(response)

     
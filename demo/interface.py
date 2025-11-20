
import sys, os
sys.dont_write_bytecode = True

import time
from dotenv import load_dotenv

import pandas as pd
import streamlit as st
import openai
from streamlit_modal import Modal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_agent import ChatBot
from ingest_data import ingest
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity


welcome_message = """
### Overview

This application helps identify relevant resumes for a given job description using a retrieval-based workflow.  
It compares your query with the uploaded resume dataset, retrieves the closest matches, and then uses an LLM to generate an analysis or summary.

### Getting Started

1. Enter your OpenAI API key.  
2. Upload a CSV file containing candidate resumes.  
3. Type a job description or question in the chat box.

The system will automatically retrieve and analyze the most relevant profiles.
"""


info_message = """
### Information

**Uploading Data**  
Upload a CSV file containing two columns: `ID` and `Resume`.  
The system will index the resume text for retrieval.

**RAG Mode**  
You can choose between basic retrieval and a fusion-based retrieval strategy.

**Data Handling**  
All data is processed locally within your current session.  
Nothing is stored externally.

**Usage**  
You may ask follow-up questions. The model uses retrieved resumes as context when appropriate.
"""

about_message = """
### About

This tool demonstrates a retrieval-augmented workflow for resume screening.  
It is intended as a technical prototype exploring how vector search and LLMs can support candidate evaluation tasks.
"""


st.title("Resume Screening GPT")

if "chat_history" not in st.session_state:
  st.session_state.chat_history = [AIMessage(content=welcome_message)]

if "embedding_model" not in st.session_state:
  st.session_state.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

if "resume_list" not in st.session_state:
  st.session_state.resume_list = []



def upload_file():
  modal = Modal(key="Demo Key", title="File Error", max_width=500)
  if st.session_state.uploaded_file != None:
    print("Uploaded file:#####################################################################################")
    try:  
      df_load = pd.read_csv(st.session_state.uploaded_file)
    except Exception as error:
      with modal.container():
        st.markdown("The uploaded file returns the following error message. Please check your csv file again.")
        st.error(error)
    else:
      if "Resume" not in df_load.columns or "ID" not in df_load.columns:
        with modal.container():
          st.error("Please include the following columns in your data: \"Resume\", \"ID\".")
      else:
        with st.toast('Indexing the uploaded data. This may take a while...'):
          st.session_state.df = df_load
          vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
          st.session_state.retriever = SelfQueryRetriever(vectordb, st.session_state.df)





def check_openai_api_key(api_key: str):
  openai.api_key = api_key
  try:
    _ = openai.chat.completions.create(
      model="gpt-4o-mini",  # Use a model you have access to
      messages=[{"role": "user", "content": "Hello!"}],
      max_tokens=3
    )
    return True
  except openai.AuthenticationError as e:
    return False
  else:
    return True
  
  
def check_model_name(model_name: str, api_key: str):
  openai.api_key = api_key
  model_list = [model.id for model in openai.models.list()]
  return True if model_name in model_list else False


def clear_message():
  st.session_state.resume_list = []
  st.session_state.chat_history = [AIMessage(content=welcome_message)]



user_query = st.chat_input("Type your message here...")

with st.sidebar:
  st.markdown("### Settings")


  st.text_input("OpenAI's API Key", type="password", key="api_key")
  st.selectbox("RAG Mode", ["Generic RAG", "RAG Fusion"], placeholder="Generic RAG", key="rag_selection")
  st.text_input("GPT Model", "gpt-4o-mini", key="gpt_selection")
  st.file_uploader("Upload resumes", type=["csv"], key="uploaded_file", on_change=upload_file)
  st.button("Clear conversation", on_click=clear_message)

  uploaded_zip = st.file_uploader("Upload Folder of Resumes (ZIP)", type=["zip"])

  st.divider()
  st.markdown(info_message)

  st.divider()
  st.markdown(about_message)

# #############################################################################
import zipfile
import tempfile
import os
import pandas as pd

from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader, PDFPlumberLoader

def process_zip(uploaded_zip):
    temp_dir = tempfile.mkdtemp()

    # Save uploaded ZIP
    zip_path = os.path.join(temp_dir, "resumes.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())

    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    records = []

    # Loop through extracted files
    for root, dirs, files in os.walk(temp_dir):
      for file in files:
        file_path = os.path.join(root, file)

        # Process ONLY valid resume file types
        if file.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        elif file.lower().endswith(".pdf"):
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()
            text = "\n".join([d.page_content for d in docs])

        elif file.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            text = docs[0].page_content

        else:
            continue  # Skip folders, system files, etc.

        resume_id = os.path.splitext(file)[0]
        print(f"Processing resume ID: {resume_id}")

        records.append({"ID": resume_id, "Resume": text})

    df = pd.DataFrame(records)
    return df

if uploaded_zip is not None:
    df_load = process_zip(uploaded_zip)
    st.session_state.df = df_load
    vectordb = ingest(df_load, "Resume", st.session_state.embedding_model)
    st.session_state.retriever = SelfQueryRetriever(vectordb, df_load)
    st.success("Successfully indexed folder of resumes!")


##########################################################################################3

for message in st.session_state.chat_history:
  if isinstance(message, AIMessage):
    with st.chat_message("AI"):
      st.write(message.content)
  elif isinstance(message, HumanMessage):
    with st.chat_message("Human"):
      st.write(message.content)
  else:
    with st.chat_message("AI"):
      message[0].render(*message[1:])


if not st.session_state.api_key:
  st.info("Please add your OpenAI API key to continue. Learn more about [API keys](https://platform.openai.com/api-keys).")
  st.stop()

if not check_openai_api_key(st.session_state.api_key):
  st.error("The API key is incorrect. Please set a valid OpenAI API key to continue. Learn more about [API keys](https://platform.openai.com/api-keys).")
  st.stop()

if not check_model_name(st.session_state.gpt_selection, st.session_state.api_key):
  st.error("The model you specified does not exist. Learn more about [OpenAI models](https://platform.openai.com/docs/models).")
  st.stop()


if "df" not in st.session_state:
    st.error("Please upload a resume CSV file to continue.")
    st.stop()

if "retriever" not in st.session_state:
    st.error("No retriever found. Upload a CSV file first.")
    st.stop()

retriever = st.session_state.retriever

llm = ChatBot(
  api_key=st.session_state.api_key,
  model=st.session_state.gpt_selection,
)

if user_query is not None and user_query != "":
  with st.chat_message("Human"):
    st.markdown(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))

  with st.chat_message("AI"):
    start = time.time()
    with st.spinner("Generating answers..."):
      document_list = retriever.retrieve_docs(user_query, llm, st.session_state.rag_selection)
      query_type = retriever.meta_data["query_type"]
      st.session_state.resume_list = document_list
      stream_message = llm.generate_message_stream(user_query, document_list, st.session_state.chat_history, query_type)
    end = time.time()

    response = st.write_stream(stream_message)
    
    retriever_message = chatbot_verbosity
    retriever_message.render(document_list, retriever.meta_data, end-start)
  
    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.chat_history.append((retriever_message, document_list, retriever.meta_data, end-start))
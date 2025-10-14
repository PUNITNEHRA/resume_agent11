import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List

# Load .env
load_dotenv()

# Make sure the token is in environment variables
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if HF_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

def get_embedding_model():
    """
    Returns a Hugging Face embedding model.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # just the model
    )

def embed_texts(texts: List[str]):
    """
    Generate embeddings for a list of texts.
    """
    emb = get_embedding_model()
    return emb.embed_documents(texts)

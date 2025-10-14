import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List

def get_embedding_model():
    # You can swap this for any Hugging Face embedding model you like
    # Examples:
    #   "sentence-transformers/all-MiniLM-L6-v2" (fast, small)
    #   "BAAI/bge-base-en-v1.5" (higher quality)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts: List[str]):
    emb = get_embedding_model()
    return emb.embed_documents(texts)


# embeddings = embed_texts(["This is a test.", "Another test sentence."])
# print(embeddings[0])
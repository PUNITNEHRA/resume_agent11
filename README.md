# Resume Screening RAG Pipeline

An AI-powered resume screening assistant built using Retrieval-Augmented Generation (RAG).  
This project demonstrates how LLMs combined with adaptive retrieval methods can analyze resumes, compare candidates, and match them with job descriptions more effectively than traditional keyword-based systems.

---

## Overview

The system processes resumes, converts them into vector embeddings, retrieves the most relevant candidates, and supplies the context to an LLM for:

- Resume summarization  
- Candidate ranking  
- Job description matching  
- Comparing multiple candidates  
- Recommendation generation  

### Key Features

- **Adaptive Retrieval**
  - *Similarity-based retrieval:* Retrieves the most relevant resume chunks using RAG or RAG Fusion.
  - *ID-based retrieval:* Directly fetches profiles when candidate IDs are provided.

- **RAG Fusion**
  Generates multiple sub-queries for complex job descriptions and merges their results for improved accuracy.

- **LLM-Powered Analysis**
  Provides structured insights using retrieved resume data.

- **Chunk-Based FAISS Indexing**
  Efficient storage and retrieval of embeddings for faster search.

### Tech Stack

- Python  
- LangChain  
- OpenAI / HuggingFace  
- FAISS  
- Streamlit  

---

## Project Structure

root/
│
├── interface.py # Streamlit application
├── rag_pipeline/ # Retrieval + LLM processing logic
├── resumes/ # Input resume text/PDFs
├── embeddings/ # FAISS vector index storage
├── requirements.txt # Package dependencies
└── README.md



---

## How to Run

1. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Linux / macOS
   venv\Scripts\activate         # Windows


pip install -r requirements.txt
streamlit run interface.py

from .chains.critique_chain import make_resume_critique_chain
from .chains.match_chain import make_match_chain
from .utils.embedding_utils import embed_texts
from .utils.vector_store import VectorStoreManager
from typing import List, Dict

class ResumeAgent:
    def __init__(self):
        self.vs_mgr = VectorStoreManager()
        self.critique_chain = make_resume_critique_chain()
        self.match_chain = make_match_chain()

    def ingest_resume(self, text: str):
        """
        Splits resume into chunks, generates embeddings using Hugging Face, 
        and adds them to the vector store.
        """
        paragraphs = [p for p in text.split("\n") if p.strip()]
        docs = [{"page_content": para, "metadata": {"chunk_id": i}} for i, para in enumerate(paragraphs)]
        self.vs_mgr.add_documents(docs)  # embeddings handled automatically
        # Generate embeddings for all paragraphs at once
        # embeddings = embed_texts([doc["page_content"] for doc in docs])
        
        # Add docs + embeddings to vector store
        # self.vs_mgr.add_documents(docs, embeddings)

    def critique(self) -> str:
        retrieved = self.vs_mgr.query("resume guidelines", k=3)
        guidelines = "\n".join([d.page_content for d in retrieved])
        resume_snips = "\n".join([d.page_content for d in self.vs_mgr.query("user resume" , k=3)])
 
        return self.critique_chain.run({
            "resume_snippets": resume_snips,
            "guideline_snippets": guidelines
        })

    def match_to_job(self, job_desc: str) -> str:
        resume_snips = "\n".join([d.page_content for d in self.vs_mgr.query("user resume" , k=3)])
        return self.match_chain.run({
            "resume_snippets": resume_snips,
            "job_desc": job_desc
        })


agent = ResumeAgent()
st = agent.ingest_resume("Experienced software developer with expertise in Python, FastAPI, and machine learning. Proven track record of building scalable web applications and deploying ML models.")
print(st)
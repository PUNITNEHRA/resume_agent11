import os
from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load environment variables from .env
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def make_resume_critique_chain():
    # ✅ Define prompt template
    prompt = PromptTemplate(
        input_variables=["resume_snippets", "guideline_snippets"],
        template="""
You are a resume expert. Given these snippets from a resume:
{resume_snippets}

And here are relevant best practices and guidelines:
{guideline_snippets}

Please critique the resume: structure issues, clarity, missing elements, phrasing, strengths & weaknesses. Provide a numbered list of suggestions.
"""
    )

    # ✅ Set up Hugging Face model endpoint with token from .env
    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=HF_TOKEN,  # token loaded securely
        repo_id="google/gemma-2-2b-it",     # public HF model
        task="conversational",
        temperature=0.7,
        max_new_tokens=256
    )

    # Wrap the HuggingFaceEndpoint with ChatHuggingFace
    chat_model = ChatHuggingFace(llm=llm)

    # ✅ Return the LLM chain
    return LLMChain(llm=chat_model, prompt=prompt)

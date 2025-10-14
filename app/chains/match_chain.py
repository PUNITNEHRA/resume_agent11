import os
from langchain import LLMChain, PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace # Import ChatHuggingFace

# ✅ Set Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def make_match_chain():
    # ✅ Define the prompt template
    prompt = PromptTemplate(
        input_variables=["resume_snippets", "job_desc"],
        template="""
You are an expert in resume-job matching. Given these resume snippets:
{resume_snippets}

And given this job description:
{job_desc}

Compare and compute a match score (0-100). Then suggest which parts of the resume to emphasize, add, or rephrase to improve the match.
"""
    )

    # ✅ Use a Hugging Face model instead of OpenAI
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-2-2b-it",  # public HF model
        task="conversational", # Changed task to conversational
        temperature=0.7,
        max_new_tokens=256
    )
    # Wrap the HuggingFaceEndpoint with ChatHuggingFace
    chat_model = ChatHuggingFace(llm=llm)


    # ✅ Return the LLM chain
    return LLMChain(llm=chat_model, prompt=prompt) # Use chat_model here

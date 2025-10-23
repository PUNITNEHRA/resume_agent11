import streamlit as st
import threading
import requests
import time
import uvicorn
from app.main import app  #  imports your FastAPI app

# Start FastAPI server in background

def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

# Start FastAPI server in a background thread
thread = threading.Thread(target=run_fastapi, daemon=True)
thread.start()

# Wait a few seconds for API to start
time.sleep(2)

## Streamlit UI
st.set_page_config(page_title="Resume Assistant Agent", page_icon="💼", layout="centered")
st.title(" Resume Assistant Agent")

# Base API URL (FastAPI is running locally)
API_BASE = "http://127.0.0.1:8000/agent"

# Tabs for different features
tab = st.sidebar.radio("Choose an option:", [" Upload Resume", "Critique Resume", " Match to Job"])

# # Upload Resum
if tab == " Upload Resume":
    st.header(" Upload your Resume File")

    uploaded_file = st.file_uploader("Upload a PDF or DOCX resume:", type=["pdf", "docx"])

    if uploaded_file is not None:
        st.info(f"Selected file: {uploaded_file.name}")

        if st.button("Ingest Resume"):
            with st.spinner("Uploading and processing your resume..."):
                try:
                    # Send file to FastAPI
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    res = requests.post(f"{API_BASE}/upload_resume/", files=files)

                    if res.status_code == 200:
                        data = res.json()
                        st.success(" Resume uploaded and processed successfully!")
                        st.text_area("Extracted Resume Text:", data.get("extracted_text", ""), height=200)
                    else:
                        st.error(f" Server Error: {res.text}")
                except Exception as e:
                    st.error(f" Could not connect to API: {e}")

# Critique Resu--
elif tab == " Critique Resume":
    st.header("Resume Critique")

    if st.button("Get Resume Critique"):
        with st.spinner("Generating critique..."):
            try:
                res = requests.post(f"{API_BASE}/critique/")
                if res.status_code == 200:
                    st.success("Critique generated!")
                    st.write(res.json().get("critique", "No critique returned."))
                else:
                    st.error(f" Error: {res.text}")
            except Exception as e:
                st.error(f"Could not connect to API: {e}")

# -------------------------------
# Match Resume to Job
elif tab == " Match to Job":
    st.header(" Match Resume to a Job Description")

    job_desc = st.text_area("Paste the job description:", height=200)

    if st.button("Match to Job"):
        with st.spinner("Analyzing job match..."):
            try:
                res = requests.post(f"{API_BASE}/match_job/", params={"job_description": job_desc})
                if res.status_code == 200:
                    st.success(" Job match analysis complete!")
                    st.write(res.json().get("match_advice", "No advice returned."))
                else:
                    st.error(f" Error: {res.text}")
            except Exception as e:
                st.error(f" Could not connect to API: {e}")

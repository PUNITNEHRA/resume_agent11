import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from .agent_runner import ResumeAgent
from .utils.doc_parser import extract_text_from_file

router = APIRouter()
agent = ResumeAgent()


@router.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...)):
    try:
        # ✅ Create a temporary file safely
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp_path = tmp.name
            # Write uploaded content to the temp file
            contents = await file.read()
            tmp.write(contents)

        # ✅ Now extract text from the file
        text = extract_text_from_file(tmp_path, file.filename)

        # ✅ (optional) remove temp file after reading
        os.remove(tmp_path)

        agent.ingest_resume(text)
        return {"message": "File uploaded and processed successfully", "extracted_text": text[:]}  # limit preview

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {e}")

# @router.post("/upload_resume/")
# async def upload_resume(file: UploadFile = File(...)):
#     # save temporarily
#     contents = await file.read()
#     tmp_path = f"/tmp/{file.filename}"
#     with open(tmp_path, "wb") as f:
#         f.write(contents)

    # try:
    #     text = extract_text_from_file(tmp_path, file.filename)
    # except Exception as e:
    #     raise HTTPException(status_code=400, detail=f"Error parsing file: {e}")

    # return {"message": "Resume ingested successfully", "num_chunks": len(text.split("\n"))}

@router.post("/critique/")
async def critique():
    result = agent.critique()
    return {"critique": result}

@router.post("/match_job/")
async def match_job(job_description: str):
    result = agent.match_to_job(job_description)
    return {"match_advice": result}



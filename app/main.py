from fastapi import FastAPI
from .api import router as api_router

app = FastAPI(title="Resume Assistant Agent")
app.include_router(api_router, prefix="/agent")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
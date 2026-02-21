# venv\Scripts\activate
#uvicorn backend.main:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
from backend.agent_system import agent
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File

import logging
from backend.logger_config import setup_logger
setup_logger()
logger = logging.getLogger(__name__)


app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Hello! RAG API is running. Use /upload_pdf to upload PDFs and /ask to query."}


@app.post("/ask")
async def ask(request: QueryRequest):
    logger.info(f"Received query : {request.query}") 

    try:
        response = await agent.run(request.query)
        logger.info(f"Agent responded successfully")
        return {"answer": str(response)}
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        return JSONResponse(status_code=500, content={"error": "An error occurred while processing the query."})

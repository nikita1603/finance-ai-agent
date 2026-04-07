# venv\Scripts\activate
# uvicorn backend.main:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
from backend.agent_system import agent
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File
from llama_index.core.agent.workflow import (
    ToolCallResult,
    AgentStream,
)

import logging
from backend.logger_config import setup_logger
setup_logger()
logger = logging.getLogger(__name__)


app = FastAPI()

class QueryRequest(BaseModel):
    query: str

query_counter = 0

@app.get("/")
def read_root():
    return {"message": "Hello! RAG API is running. Use /upload_pdf to upload PDFs and /ask to query."}


@app.post("/ask")
async def ask(request: QueryRequest):
    global query_counter
    logger.info(f"Received query : {request.query}") 
    query_counter += 1

    try:
        handler = agent.run(request.query)
        async for event in handler.stream_events():
            if isinstance(event, AgentStream):
                logger.info(event.delta)
            elif isinstance(event, ToolCallResult):
                logger.info(f"EVALUATION_LOGS | TOOL_CALL | QUERY_NUM: {query_counter} | TOOL: {event.tool_name}")
                logger.info(f"TOOL CALL DONE | TOOL: {event.tool_name} | ARGS: {event.tool_kwargs} | RESULT: {event.tool_output}")   
        response = await handler
        logger.info(f"Agent responded successfully")
        logger.info(f"EVALUATION_LOGS | QUERY_RESPONSE | QUERY_NUM: {query_counter} | RESPONSE: {str(response).replace('\n', ' ')}")
        return {"answer": str(response)}
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        return JSONResponse(status_code=500, content={"error": "An error occurred while processing the query."})

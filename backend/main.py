from fastapi import FastAPI
from pydantic import BaseModel
from backend.agent_system import agent
from fastapi.responses import JSONResponse
from fastapi import FastAPI
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
    """Pydantic model for the `/ask` endpoint payload.

    Attributes:
        query: str - the user's natural language query.
    """
    query: str

query_counter = 0

@app.get("/")
def read_root():
    """Health/info endpoint.

    Returns a short message describing available endpoints.
    """
    return {"message": "Hello! RAG API is running. Use /upload_pdf to upload PDFs and /ask to query."}


@app.post("/ask")
async def ask(request: QueryRequest):
    """Run the agent for the given query and return its final response.

    This function streams intermediate events emitted by the agent to the
    logger so partial outputs and tool call results are recorded. The
    final agent output is returned under the `answer` key.
    """
    global query_counter
    # Log the received query (keeps original log message format)
    logger.info(f"Received query : {request.query}") 
    query_counter += 1

    try:
        # Start the agent; the returned handler supports streaming events
        handler = agent.run(request.query)

        # Stream intermediate events for monitoring / evaluation
        async for event in handler.stream_events():
            if isinstance(event, AgentStream):
                # Partial text delta from the agent's generation
                logger.info(event.delta)
            elif isinstance(event, ToolCallResult):
                # Log tool invocation and result for traceability
                logger.info(f"EVALUATION_LOGS | TOOL_CALL | QUERY_NUM: {query_counter} | TOOL: {event.tool_name}")
                logger.info(f"TOOL CALL DONE | TOOL: {event.tool_name} | ARGS: {event.tool_kwargs} | RESULT: {event.tool_output}")   
        # Await the final response from the agent handler
        response = await handler
        logger.info(f"Agent responded successfully")
        # Flatten newlines for concise single-line log entry
        logger.info(f"EVALUATION_LOGS | QUERY_RESPONSE | QUERY_NUM: {query_counter} | RESPONSE: {str(response).replace('\n', ' ')}")
        return {"answer": str(response)}
    except Exception as e:
        # Preserve original error handling while capturing traceback
        logger.error(f"Error in ask endpoint: {e}")
        return JSONResponse(status_code=500, content={"error": "An error occurred while processing the query."})

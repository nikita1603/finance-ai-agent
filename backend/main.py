from fastapi import FastAPI
from pydantic import BaseModel
from backend.agent_system import agent
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from llama_index.core.agent.workflow import (
    ToolCallResult,
    AgentStream,
)

import asyncio
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

    max_retries = 4
    tool_outputs: dict[str, str] = {}
    for attempt in range(max_retries):
        try:
            handler = agent.run(request.query)

            async for event in handler.stream_events():
                if isinstance(event, AgentStream):
                    logger.info(event.delta)
                elif isinstance(event, ToolCallResult):
                    logger.info(f"EVALUATION_LOGS | TOOL_CALL | QUERY_NUM: {query_counter} | TOOL: {event.tool_name}")
                    logger.info(f"TOOL CALL DONE | TOOL: {event.tool_name} | ARGS: {event.tool_kwargs} | RESULT: {event.tool_output}")
                    tool_outputs[event.tool_name] = str(event.tool_output)

            response = await handler
            logger.info(f"Agent responded successfully")
            logger.info(f"EVALUATION_LOGS | QUERY_RESPONSE | QUERY_NUM: {query_counter} | RESPONSE: {str(response).replace('\n', ' ')}")
            return {"answer": str(response)}

        except Exception as e:
            err_str = str(e)
            is_last = attempt == max_retries - 1
            is_retryable = "503" in err_str or "429" in err_str
            if not is_retryable or is_last:
                logger.error(f"Error in ask endpoint: {e}")
                if tool_outputs:
                    fallback = "\n\n---\n".join(
                        f"[{name} output]\n{output}" for name, output in tool_outputs.items()
                    )
                    logger.info("Returning tool outputs as fallback due to agent LLM failure")
                    return {"answer": f"Agent LLM unavailable ({e}). Showing raw tool output:\n\n{fallback}"}
                return JSONResponse(status_code=500, content={"error": "An error occurred while processing the query."})
            # For 429, parse the suggested retry delay from the error message; fall back to exponential backoff
            delay = 2 ** attempt
            if "429" in err_str:
                import re
                match = re.search(r"retry.*?(\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                if match:
                    delay = float(match.group(1)) + 2  # small buffer
            label = "429 rate-limit" if "429" in err_str else "503"
            logger.warning(f"Agent {label} on attempt {attempt + 1}/{max_retries}, retrying in {delay}s: {e}")
            await asyncio.sleep(delay)

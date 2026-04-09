"""
Ollama client service.

Encapsulates the HTTP interaction with a local Ollama server so the rest of the
app (routers, schemas) does not need to know about httpx, URL construction, or
Ollama's response format. Errors are translated into FastAPI HTTPExceptions so
routers can simply call `chat_completion(...)` and let exceptions propagate.
"""

import httpx
from fastapi import HTTPException

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from schemas.chat import ChatMessage


async def chat_completion(messages: list[ChatMessage]) -> tuple[str, str]:
    """
    Send a conversation to Ollama's /api/chat endpoint and return the reply.

    Streaming is disabled so a single consolidated response is returned.

    Args:
        messages: Conversation history to send to the model.

    Returns:
        A tuple of (model_name_used, assistant_reply_content).

    Raises:
        HTTPException: 502 if Ollama is unreachable or returns an error,
                       504 if the request to Ollama times out.
    """
    model_name = OLLAMA_MODEL
    payload = {
        "model": model_name,
        "messages": [m.model_dump() for m in messages],
        "stream": False,
    }

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Ollama request timed out: {exc}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach Ollama at {OLLAMA_BASE_URL}: {exc}",
        ) from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned {response.status_code}: {response.text}",
        )

    data = response.json()
    message = data.get("message") or {}
    content = message.get("content")
    if not content:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama response missing message content: {data}",
        )

    return data.get("model", model_name), content
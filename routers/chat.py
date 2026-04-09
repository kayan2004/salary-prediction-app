"""
Chat router — exposes the /chat endpoint.

This router is intentionally thin: it only wires HTTP requests to the Ollama
service. The pieces it depends on live in their own modules so each file has
a single responsibility:

    app/config.py          — environment configuration (OLLAMA_BASE_URL, OLLAMA_MODEL)
    app/schemas/chat.py    — Pydantic request/response models
    app/services/ollama.py — HTTP client logic for talking to Ollama

Endpoints:
    POST /chat   — Send a list of messages and get a single assistant reply back.
"""

from fastapi import APIRouter

from schemas.chat import ChatMessage, ChatRequest, ChatResponse
from services.ollama import chat_completion

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a conversation to the local Ollama model and return its reply.

    Args:
        request (ChatRequest): The conversation history to send to the model.

    Returns:
        ChatResponse: The model name used and the assistant's reply message.

    Raises:
        HTTPException: 502 if Ollama is unreachable or returns an error,
                       504 if the request to Ollama times out.
    """
    model_used, content = await chat_completion(messages=request.messages)
    return ChatResponse(
        model=model_used,
        message=ChatMessage(role="assistant", content=content),
    )
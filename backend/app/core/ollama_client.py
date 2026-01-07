from fastapi import Request
from ollama import AsyncClient

from config import settings


async def init_ollama_client():
    return AsyncClient(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {settings.ollama_api_key}"}
    )


def get_ollama_client(request: Request) -> AsyncClient:
    return request.app.state.ollama_client

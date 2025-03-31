import pytest
from httpx import AsyncClient
from main import app
from prisma import Prisma


@pytest.fixture(scope="module", autouse=True)
async def prisma_client():
    db = Prisma()
    await db.connect()
    yield db
    await db.disconnect()


@pytest.mark.asyncio
async def test_ask_route(prisma_client):
    async with AsyncClient(app=app, base_url="http://testserver") as ac:
        test_payload = {
            "query": "What is diabetes?",
            "user_id": "test-user",
            "chat_id": "test-chat",
            "top_k": 3,
        }

        response = await ac.post("/ask", json=test_payload)
        assert response.status_code in (200, 400, 403)  # Expected responses
        data = response.json()

        if response.status_code == 200:
            assert "query" in data
            assert "answer" in data
            assert data["query"] == test_payload["query"]
            assert isinstance(data["answer"], str)
        else:
            assert "detail" in data or "error" in data

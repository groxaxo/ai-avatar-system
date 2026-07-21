import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    """Test the root endpoint returns API info."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "AI Avatar System API"
    assert data["version"] == "2.0.0"
    assert data["status"] == "running"


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """Test the health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_docs_availability_matches_debug_setting(client: AsyncClient):
    """OpenAPI docs are exposed only when DEBUG is enabled."""
    from app.config import settings

    response = await client.get("/docs")
    assert response.status_code in ([200, 307] if settings.DEBUG else [404])

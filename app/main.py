from fastapi import FastAPI

from app.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI application.

    No module-level app instance is created; run with
    ``uvicorn app.main:create_app --factory``.
    """
    settings = settings or Settings()
    application = FastAPI()

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": settings.openai_model}

    return application

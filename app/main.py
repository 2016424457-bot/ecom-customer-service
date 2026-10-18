import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import Settings
from app.errors import AppError
from app.runtime import Runtime, build_runtime


def create_app(
    *,
    settings: Settings | None = None,
    runtime: Runtime | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    No module-level app instance is created; run with
    ``uvicorn app.main:create_app --factory``.
    """
    runtime = runtime or build_runtime(settings or Settings())
    logging.basicConfig(level=runtime.settings.log_level.upper())

    application = FastAPI()
    application.state.runtime = runtime
    application.include_router(router)

    @application.exception_handler(AppError)
    async def handle_app_error(
        _request: Request, exc: AppError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message},
        )

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": runtime.settings.openai_model}

    return application

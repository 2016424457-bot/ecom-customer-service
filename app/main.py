import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.actions import router as actions_router
from app.api.routes import router
from app.config import Settings
from app.errors import AppError
from app.graph import runtime as graph_runtime
from app.runtime import Runtime, build_runtime

STATIC_DIR = Path(__file__).resolve().parent / "static"


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

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        await graph_runtime.init_graph(runtime.settings)
        yield
        await graph_runtime.close_graph()
        await runtime.aclose()

    application = FastAPI(lifespan=lifespan)
    application.state.runtime = runtime
    application.mount(
        "/static", StaticFiles(directory=STATIC_DIR), name="static"
    )
    application.include_router(router)
    application.include_router(actions_router)

    @application.get("/", include_in_schema=False)
    async def index() -> RedirectResponse:
        return RedirectResponse(url="/chat")

    @application.get("/chat", include_in_schema=False)
    async def chat_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "chat.html")

    @application.get("/faith-cases", include_in_schema=False)
    async def faith_cases_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "faith-cases.html")

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

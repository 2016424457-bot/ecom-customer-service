import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.actions import router as actions_router
from app.api.conversations import router as conversations_router
from app.api.eval import router as eval_router
from app.api.review import router as review_router
from app.api.routes import router
from app.auth import (
    ensure_auth_schema,
    router as auth_router,
    session_user,
)
from app.config import Settings
from app.context import format_context_budget
from app.errors import AppError
from app.graph import runtime as graph_runtime
from app.runtime import Runtime, build_runtime

STATIC_DIR = Path(__file__).resolve().parent / "static"
logger = logging.getLogger(__name__)


def _configure_logging(settings: Settings) -> None:
    level = getattr(settings, "log_level", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    root = logging.getLogger()
    root.setLevel(level)
    log_file = getattr(settings, "log_file", None)
    if log_file is None:
        return
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    target = str(log_path.resolve())
    for handler in root.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == target
        ):
            return
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
    )
    root.addHandler(handler)


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
    _configure_logging(runtime.settings)
    budget = getattr(runtime, "context_budget", None)
    if budget is not None:
        logger.info(format_context_budget(budget))
        if not budget.ok:
            logger.error(budget.warning)
        elif budget.warning:
            logger.warning(budget.warning)

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        await graph_runtime.init_graph(runtime.settings)
        await runtime.start()
        await ensure_auth_schema(runtime)
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
    application.include_router(conversations_router)
    application.include_router(review_router)
    application.include_router(eval_router)
    application.include_router(auth_router)

    async def _current_user(request: Request):
        return await session_user(request, request.app.state.runtime)

    def _login_redirect(next_path: str, admin: bool = False):
        target = "/admin/login" if admin else "/login"
        return RedirectResponse(
            f"{target}?next={next_path}", status_code=303
        )

    @application.get("/", include_in_schema=False)
    async def index(request: Request) -> RedirectResponse:
        user = await _current_user(request)
        if user is None:
            return RedirectResponse(url="/login", status_code=303)
        if user.role == "admin":
            return RedirectResponse(url="/review-queue", status_code=303)
        return RedirectResponse(url="/chat", status_code=303)

    @application.get("/login", include_in_schema=False)
    async def login_page(request: Request):
        user = await _current_user(request)
        if user is not None:
            target = "/review-queue" if user.role == "admin" else "/chat"
            return RedirectResponse(url=target, status_code=303)
        return FileResponse(STATIC_DIR / "login.html")

    @application.get("/register", include_in_schema=False)
    async def register_page(request: Request):
        user = await _current_user(request)
        if user is not None:
            target = "/review-queue" if user.role == "admin" else "/chat"
            return RedirectResponse(url=target, status_code=303)
        return FileResponse(STATIC_DIR / "register.html")

    @application.get("/admin/login", include_in_schema=False)
    async def admin_login_page(request: Request):
        user = await _current_user(request)
        if user is not None and user.role == "admin":
            return RedirectResponse(url="/review-queue", status_code=303)
        return FileResponse(STATIC_DIR / "admin-login.html")

    @application.get("/chat", include_in_schema=False)
    async def chat_page(request: Request):
        user = await _current_user(request)
        if user is None:
            return _login_redirect("/chat")
        if user.role != "user":
            return RedirectResponse(url="/review-queue", status_code=303)
        return FileResponse(STATIC_DIR / "chat.html")

    @application.get("/review-queue", include_in_schema=False)
    async def review_queue_page(request: Request):
        user = await _current_user(request)
        if user is None:
            return _login_redirect("/review-queue", admin=True)
        if user.role != "admin":
            return RedirectResponse(url="/chat", status_code=303)
        return FileResponse(STATIC_DIR / "review-queue.html")

    @application.get("/faith-cases", include_in_schema=False)
    async def faith_cases_page(request: Request):
        user = await _current_user(request)
        if user is None:
            return _login_redirect("/faith-cases", admin=True)
        if user.role != "admin":
            return RedirectResponse(url="/chat", status_code=303)
        return FileResponse(STATIC_DIR / "faith-cases.html")

    @application.get("/flywheel", include_in_schema=False)
    async def flywheel_page(request: Request):
        user = await _current_user(request)
        if user is None:
            return _login_redirect("/flywheel", admin=True)
        if user.role != "admin":
            return RedirectResponse(url="/chat", status_code=303)
        return FileResponse(STATIC_DIR / "flywheel.html")

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

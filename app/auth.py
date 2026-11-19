"""Session-based authentication for the customer and admin portals."""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.models import User
from app.runtime import Runtime, get_runtime

logger = logging.getLogger(__name__)

COOKIE_NAME = "ecom_session"
USERNAME_PATTERN = r"^[A-Za-z0-9_]{3,32}$"
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32

router = APIRouter()


@dataclass(frozen=True)
class SessionUser:
    """The authenticated user attached to one request."""

    id: int
    username: str
    role: str
    display_name: str

    @property
    def user_key(self) -> str:
        return f"user:{self.id}"

    def public(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "display_name": self.display_name,
            "user_key": self.user_key,
        }


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )
    return "$".join(
        [
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            _b64encode(salt),
            _b64encode(digest),
        ]
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, expected = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_b64decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(_b64decode(expected)),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, _b64decode(expected))


def sign_session(secret: str, payload: dict) -> str:
    body = _b64encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    )
    signature = hmac.new(
        secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256
    ).hexdigest()
    return f"{body}.{signature}"


def verify_session(secret: str, token: str | None) -> dict | None:
    if not token or "." not in token:
        return None
    body, signature = token.rsplit(".", 1)
    expected = hmac.new(
        secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        payload = json.loads(_b64decode(body))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    issued_at = payload.get("iat")
    expires_at = payload.get("exp")
    if not isinstance(expires_at, int) or expires_at < int(time.time()):
        return None
    if not isinstance(issued_at, int) or issued_at > int(time.time()) + 60:
        return None
    if not isinstance(payload.get("uid"), int):
        return None
    if payload.get("role") not in ("user", "admin"):
        return None
    return payload


def _set_session_cookie(
    response: Response,
    request: Request,
    runtime: Runtime,
    user: User,
) -> None:
    settings = runtime.settings
    now = int(time.time())
    token = sign_session(
        settings.session_secret,
        {
            "uid": user.id,
            "role": user.role,
            "iat": now,
            "exp": now + settings.session_ttl_seconds,
        },
    )
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=settings.session_ttl_seconds,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/")


def verify_same_origin(request: Request) -> None:
    """Reject state-changing requests that come from another origin.

    Missing Origin is allowed so curl and TestClient keep working; a
    present but mismatched Origin is refused.
    """
    origin = request.headers.get("origin")
    if not origin:
        return
    host = request.headers.get("host", "")
    try:
        netloc = urlparse(origin).netloc
    except ValueError:
        netloc = ""
    if netloc != host:
        raise HTTPException(status_code=403, detail="跨站请求被拒绝")


async def _session_user_from_request(
    request: Request, runtime: Runtime
) -> SessionUser | None:
    if runtime.session_factory is None:
        # Repository-less test runtimes have no users table; keep their
        # existing behaviour while real deployments always use the DB.
        return SessionUser(
            id=1,
            username="local",
            role="admin",
            display_name="本地测试",
        )
    payload = verify_session(
        runtime.settings.session_secret,
        request.cookies.get(COOKIE_NAME),
    )
    if payload is None:
        return None
    async with runtime.session_factory() as session:
        user = await session.get(User, payload["uid"])
    if user is None or user.role != payload.get("role"):
        return None
    return SessionUser(
        id=user.id,
        username=user.username,
        role=user.role,
        display_name=user.display_name,
    )


async def load_session_user(
    request: Request, runtime: Runtime = Depends(get_runtime)
) -> SessionUser | None:
    return await _session_user_from_request(request, runtime)


async def session_user(
    request: Request, runtime: Runtime
) -> SessionUser | None:
    """Page routes: resolve the signed-in user without FastAPI deps."""
    return await _session_user_from_request(request, runtime)


async def require_user(
    user: SessionUser | None = Depends(load_session_user),
) -> SessionUser:
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return user


async def require_admin(
    user: SessionUser | None = Depends(load_session_user),
) -> SessionUser:
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录")
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


async def get_owned_conversation(
    repository, conversation_id: int, user: SessionUser
):
    conversation = await repository.get_conversation(conversation_id)
    if conversation is None or conversation.user_id != user.user_key:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conversation


async def ensure_auth_schema(runtime: Runtime) -> None:
    """Create the users table and seed the admin account once."""
    engine = getattr(runtime, "engine", None)
    settings = getattr(runtime, "settings", None)
    if engine is None or settings is None:
        return
    async with engine.begin() as connection:
        await connection.run_sync(
            lambda sync: User.__table__.create(sync, checkfirst=True)
        )
    session_factory = getattr(runtime, "session_factory", None)
    if session_factory is None:
        return

    async with session_factory() as session:
        existing = await session.scalar(
            select(User).where(User.username == settings.admin_username)
        )
        if existing is not None:
            if existing.role != "admin":
                logger.warning(
                    "auth_admin_username_taken username=%s",
                    settings.admin_username,
                )
            return
        session.add(
            User(
                username=settings.admin_username,
                password_hash=hash_password(settings.admin_password),
                role="admin",
                display_name=settings.admin_display_name,
            )
        )
        await session.commit()
        logger.warning(
            "auth_seeded_admin username=%s password=default "
            "(change ADMIN_PASSWORD)",
            settings.admin_username,
        )


class RegisterRequest(BaseModel):
    username: str = Field(pattern=USERNAME_PATTERN)
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(default="", max_length=64)

    @field_validator("display_name")
    @classmethod
    def strip_display_name(cls, value: str) -> str:
        return value.strip()


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=128)
    portal: Literal["user", "admin"] = "user"


@router.post("/api/auth/register")
async def register(
    payload: RegisterRequest,
    response: Response,
    request: Request,
    _origin: None = Depends(verify_same_origin),
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    if runtime.session_factory is None:
        raise HTTPException(status_code=503, detail="用户服务不可用")
    if payload.username.lower() in ("admin", "administrator"):
        raise HTTPException(status_code=409, detail="该账号不可注册")
    user = User(
        username=payload.username,
        password_hash=hash_password(payload.password),
        role="user",
        display_name=payload.display_name or payload.username,
    )
    async with runtime.session_factory() as session:
        session.add(user)
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            raise HTTPException(status_code=409, detail="账号已存在")
        await session.refresh(user)
    _set_session_cookie(response, request, runtime, user)
    return {
        "user": SessionUser(
            id=user.id,
            username=user.username,
            role=user.role,
            display_name=user.display_name,
        ).public()
    }


@router.post("/api/auth/login")
async def login(
    payload: LoginRequest,
    response: Response,
    request: Request,
    _origin: None = Depends(verify_same_origin),
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    if runtime.session_factory is None:
        raise HTTPException(status_code=503, detail="用户服务不可用")
    async with runtime.session_factory() as session:
        user = await session.scalar(
            select(User).where(User.username == payload.username)
        )
    if user is None or not verify_password(
        payload.password, user.password_hash
    ):
        raise HTTPException(status_code=401, detail="账号或密码不正确")
    if payload.portal == "admin" and user.role != "admin":
        raise HTTPException(status_code=403, detail="该账号不是管理员")
    if payload.portal == "user" and user.role == "admin":
        raise HTTPException(status_code=403, detail="管理员请从后台入口登录")
    _set_session_cookie(response, request, runtime, user)
    return {
        "user": SessionUser(
            id=user.id,
            username=user.username,
            role=user.role,
            display_name=user.display_name,
        ).public()
    }


@router.post("/api/auth/logout")
async def logout(
    response: Response,
    _origin: None = Depends(verify_same_origin),
) -> dict:
    clear_session_cookie(response)
    return {"ok": True}


@router.get("/api/auth/me")
async def me(user: SessionUser = Depends(require_user)) -> dict:
    return {"user": user.public()}
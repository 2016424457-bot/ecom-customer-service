import time

from app.auth import (
    hash_password,
    sign_session,
    verify_password,
    verify_session,
)


def test_password_hash_roundtrip():
    stored = hash_password("correct-horse")
    assert stored.startswith("scrypt$")
    assert verify_password("correct-horse", stored)
    assert not verify_password("wrong-horse", stored)


def test_password_hash_uses_unique_salts():
    first = hash_password("same-password")
    second = hash_password("same-password")
    assert first != second
    assert verify_password("same-password", first)
    assert verify_password("same-password", second)


def test_session_roundtrip():
    payload = {"uid": 7, "role": "user", "iat": int(time.time())}
    payload["exp"] = payload["iat"] + 3600
    token = sign_session("secret", payload)
    decoded = verify_session("secret", token)
    assert decoded is not None
    assert decoded["uid"] == 7
    assert decoded["role"] == "user"


def test_session_rejects_tampering():
    payload = {"uid": 7, "role": "user", "iat": 1, "exp": 2**31}
    token = sign_session("secret", payload)
    body, signature = token.rsplit(".", 1)
    forged = f"{body}.{'0' * len(signature)}"
    assert verify_session("secret", forged) is None
    assert verify_session("other-secret", token) is None


def test_session_rejects_expired():
    now = int(time.time())
    token = sign_session(
        "secret",
        {"uid": 7, "role": "user", "iat": now - 10, "exp": now - 1},
    )
    assert verify_session("secret", token) is None
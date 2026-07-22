"""JWT session tokens for the new SQL-backed auth path.

Replaces backend/api.py's hand-rolled HMAC token (`_create_token`/
`_read_token`, api.py:100-136) once PR6 flips AUTH_BACKEND to "jwt". The two
are not wire-compatible -- deploying this is a hard cutover, every existing
session is invalidated at once (see PR6 in the implementation plan).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import jwt as _pyjwt

from backend.config import jwt_secret_key

ALGORITHM = "HS256"
_DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # matches the legacy token's 7-day TTL


class TokenError(Exception):
    """Raised for any invalid/expired/malformed token -- callers (backend/auth/
    dependencies.py) catch this one type rather than depending on PyJWT's
    exception hierarchy directly."""


@dataclass(frozen=True)
class TokenPayload:
    account_id: str
    account_kind: str
    jti: str
    issued_at: int
    expires_at: int


def create_access_token(account_id: str, account_kind: str, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> str:
    now = int(time.time())
    claims = {
        "sub": account_id,
        "kind": account_kind,
        "iat": now,
        "exp": now + ttl_seconds,
        "jti": uuid.uuid4().hex,
    }
    return _pyjwt.encode(claims, jwt_secret_key(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> TokenPayload:
    try:
        claims = _pyjwt.decode(token, jwt_secret_key(), algorithms=[ALGORITHM])
    except _pyjwt.PyJWTError as exc:
        raise TokenError(str(exc)) from exc

    account_id = claims.get("sub")
    account_kind = claims.get("kind")
    if not account_id or not account_kind:
        raise TokenError("Token is missing required claims.")

    return TokenPayload(
        account_id=str(account_id),
        account_kind=str(account_kind),
        jti=str(claims.get("jti", "")),
        issued_at=int(claims.get("iat", 0)),
        expires_at=int(claims.get("exp", 0)),
    )

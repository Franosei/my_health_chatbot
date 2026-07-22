"""Password hashing for the new SQL-backed accounts table.

Two algorithms coexist by design during the migration window:

- "argon2id": every account created or rotated after the SQL cutover.
- "pbkdf2_sha256_legacy": accounts carried over verbatim by
  backend/scripts/migrate_json_to_sql.py from the old HMAC/PBKDF2 store
  (backend/user_store.py's `_hash_password` -- 200,000 iterations,
  hex-encoded salt+hash) so migration doesn't force a mass password reset.
  `needs_rehash` flags these so the caller (backend/auth/dependencies.py,
  once wired in PR6) can transparently upgrade an account to argon2id the
  next time that password is verified successfully.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

ARGON2 = "argon2id"
PBKDF2_SHA256_LEGACY = "pbkdf2_sha256_legacy"

_hasher = PasswordHasher()


@dataclass(frozen=True)
class PasswordHash:
    algo: str
    hash: str
    salt: Optional[str] = None  # only set for pbkdf2_sha256_legacy


def hash_password(password: str) -> PasswordHash:
    return PasswordHash(algo=ARGON2, hash=_hasher.hash(password), salt=None)


def _verify_legacy_pbkdf2(password: str, password_hash: str, salt: str) -> bool:
    computed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000)
    return computed.hex() == password_hash


def verify_password(password: str, password_hash: str, password_algo: str, password_salt: Optional[str] = None) -> bool:
    if password_algo == ARGON2:
        try:
            return _hasher.verify(password_hash, password)
        except VerifyMismatchError:
            return False
        except Exception:
            return False
    if password_algo == PBKDF2_SHA256_LEGACY:
        if not password_salt:
            return False
        return _verify_legacy_pbkdf2(password, password_hash, password_salt)
    return False


def needs_rehash(password_algo: str) -> bool:
    return password_algo != ARGON2

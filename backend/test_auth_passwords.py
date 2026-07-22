import hashlib

from backend.auth.passwords import (
    ARGON2,
    PBKDF2_SHA256_LEGACY,
    hash_password,
    needs_rehash,
    verify_password,
)


def test_hash_password_returns_argon2id_and_verifies():
    hashed = hash_password("correct horse battery staple")
    assert hashed.algo == ARGON2
    assert hashed.salt is None
    assert verify_password("correct horse battery staple", hashed.hash, hashed.algo)


def test_verify_password_rejects_wrong_password():
    hashed = hash_password("correct horse battery staple")
    assert not verify_password("wrong password", hashed.hash, hashed.algo)


def test_verify_legacy_pbkdf2_matches_user_store_algorithm():
    # Mirrors backend/user_store.py's _hash_password exactly, so migrated
    # accounts (backend/scripts/migrate_json_to_sql.py) keep working.
    password = "legacy-password-123"
    salt = "abc123salt"
    legacy_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000).hex()

    assert verify_password(password, legacy_hash, PBKDF2_SHA256_LEGACY, salt)
    assert not verify_password("wrong", legacy_hash, PBKDF2_SHA256_LEGACY, salt)


def test_verify_legacy_pbkdf2_without_salt_fails_closed():
    assert not verify_password("anything", "somehash", PBKDF2_SHA256_LEGACY, None)


def test_verify_password_unknown_algo_fails_closed():
    assert not verify_password("anything", "somehash", "made-up-algo")


def test_needs_rehash():
    assert needs_rehash(PBKDF2_SHA256_LEGACY) is True
    assert needs_rehash(ARGON2) is False

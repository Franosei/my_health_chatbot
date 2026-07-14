import pytest

from evaluations import retry


def test_retries_transient_failure_then_succeeds(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda *_: None)
    attempts = 0

    class TemporaryError(Exception):
        status_code = 503

    def typed_flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TemporaryError()
        return "ok"

    attempts = 0
    assert retry.call_with_retry(typed_flaky, base_delay=0) == "ok"
    assert attempts == 3


def test_does_not_retry_non_retryable_authorization_error(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda *_: None)

    class AuthorizationError(Exception):
        status_code = 401

    attempts = 0

    def fail():
        nonlocal attempts
        attempts += 1
        raise AuthorizationError()

    with pytest.raises(AuthorizationError):
        retry.call_with_retry(fail)
    assert attempts == 1

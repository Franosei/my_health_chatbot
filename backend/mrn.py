"""Patient MRN (Medical Record Number) generation and validation.

MRNs are the durable, clinician-facing patient identifier -- distinct from
username/email (which patients may change) and from the internal database
primary key (which should never be exposed outside the system). They are:

- Random, not sequential or derived from any PII (name, DOB, email) --
  sequential/derivable IDs make patient records enumerable.
- Fixed-length with a trailing check symbol, so a mistyped/mistranscribed
  MRN is caught client-side before it ever becomes a lookup query, and a
  guessed ID is unlikely to pass the checksum.
- Formatted for a human to read aloud or write on paper: `FM-XXXX-XXXX`.

The alphabet excludes visually ambiguous characters (I, L, O, U), following
Crockford's base32 convention, though the check symbol here is a simple
weighted mod-32 digit -- not full Crockford base32 check-symbol compliance.
"""

from __future__ import annotations

import secrets

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32 alphabet (32 symbols)
_GROUP_SIZE = 4
_PAYLOAD_GROUPS = 2
_PREFIX = "FM"


def _checksum(payload: str) -> str:
    total = 0
    for index, char in enumerate(payload):
        total += _ALPHABET.index(char) * (index + 1)
    return _ALPHABET[total % len(_ALPHABET)]


def generate_mrn() -> str:
    """Generate a new, random, checksum-bearing patient MRN."""
    payload_len = _GROUP_SIZE * _PAYLOAD_GROUPS - 1  # last symbol is the check digit
    payload = "".join(secrets.choice(_ALPHABET) for _ in range(payload_len))
    payload += _checksum(payload)

    groups = [payload[i : i + _GROUP_SIZE] for i in range(0, len(payload), _GROUP_SIZE)]
    return f"{_PREFIX}-{'-'.join(groups)}"


def normalize_mrn(raw: str) -> str:
    """Uppercase and strip formatting so lookups are forgiving of dashes/case."""
    return "".join(ch for ch in (raw or "").strip().upper() if ch.isalnum())


def is_valid_mrn(raw: str) -> bool:
    """Structural + checksum validation -- does not check whether the MRN is assigned."""
    normalized = normalize_mrn(raw)
    if not normalized.startswith(_PREFIX):
        return False
    payload = normalized[len(_PREFIX) :]
    if len(payload) != _GROUP_SIZE * _PAYLOAD_GROUPS:
        return False
    if any(ch not in _ALPHABET for ch in payload):
        return False
    body, check_digit = payload[:-1], payload[-1]
    return _checksum(body) == check_digit

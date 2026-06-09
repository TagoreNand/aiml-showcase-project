"""Minimal dependency-free JWT (HS256) encode/decode.

Implemented with the standard library so the core app needs no extra packages.
In production you can swap this for PyJWT without changing call sites.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict


class JWTError(Exception):
    pass


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(seg: str) -> bytes:
    pad = "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg + pad)


def encode(payload: Dict[str, Any], secret: str, *, algorithm: str = "HS256",
           expires_minutes: int = 60) -> str:
    if algorithm != "HS256":
        raise JWTError(f"unsupported algorithm: {algorithm}")
    header = {"alg": algorithm, "typ": "JWT"}
    now = int(time.time())
    body = {**payload, "iat": now, "exp": now + expires_minutes * 60}
    seg = [
        _b64url_encode(json.dumps(header, separators=(",", ":")).encode()),
        _b64url_encode(json.dumps(body, separators=(",", ":")).encode()),
    ]
    signing_input = ".".join(seg).encode()
    sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return signing_input.decode() + "." + _b64url_encode(sig)


def decode(token: str, secret: str, *, algorithm: str = "HS256") -> Dict[str, Any]:
    try:
        header_seg, payload_seg, sig_seg = token.split(".")
    except ValueError as exc:
        raise JWTError("malformed token") from exc
    signing_input = f"{header_seg}.{payload_seg}".encode()
    expected = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _b64url_decode(sig_seg)):
        raise JWTError("signature mismatch")
    payload = json.loads(_b64url_decode(payload_seg))
    if "exp" in payload and int(time.time()) > int(payload["exp"]):
        raise JWTError("token expired")
    return payload

"""Google Sign-In: verifies Google ID tokens and keeps the user in a signed session cookie."""

import logging
import os
import secrets
from typing import Optional

from fastapi import Request, Response
from google.auth.transport import requests
from google.oauth2 import id_token
from itsdangerous import BadSignature, URLSafeTimedSerializer

logger = logging.getLogger(__name__)

COOKIE_NAME = "hunter_session"
SESSION_MAX_AGE = int(os.getenv("SESSION_MAX_AGE_HOURS", "168")) * 3600  # default 7 days
_GOOGLE_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}

_fallback_secret: Optional[str] = None


def client_id() -> str:
    # Tolerate stray whitespace or quotes pasted into the Cloud Run variable.
    return os.getenv("GOOGLE_CLIENT_ID", "").strip().strip("'\"").strip()


def enabled() -> bool:
    return bool(client_id())


def is_allowed(email: str) -> bool:
    allowed = os.getenv("ALLOWED_EMAILS", "")
    if not allowed:
        return True
    email = (email or "").lower()
    allowed_list = [e.strip().lower() for e in allowed.split(",") if e.strip()]
    # "@example.com" allows a whole domain
    return any(email == a or (a.startswith("@") and email.endswith(a)) for a in allowed_list)


def _serializer() -> URLSafeTimedSerializer:
    global _fallback_secret
    secret = os.getenv("SESSION_SECRET")
    if not secret:
        if _fallback_secret is None:
            logger.warning(
                "SESSION_SECRET is not set; using a random per-process secret. "
                "Sessions will not survive restarts or work across instances."
            )
            _fallback_secret = secrets.token_urlsafe(32)
        secret = _fallback_secret
    return URLSafeTimedSerializer(secret, salt="hunter-session")


def verify_google_credential(credential: str) -> dict:
    cid = client_id()
    if not cid:
        raise ValueError("GOOGLE_CLIENT_ID is not configured")
    id_info = id_token.verify_oauth2_token(credential, requests.Request(), cid)
    if id_info.get("iss") not in _GOOGLE_ISSUERS:
        raise ValueError("Wrong token issuer")
    if not id_info.get("email") or not id_info.get("email_verified"):
        raise ValueError("Google account email is not verified")
    return {
        "email": id_info["email"],
        "name": id_info.get("name") or id_info["email"],
        "picture": id_info.get("picture"),
    }


def start_session(response: Response, user: dict):
    response.set_cookie(
        key=COOKIE_NAME,
        value=_serializer().dumps(user),
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=os.getenv("SESSION_COOKIE_SECURE", "true").lower() not in {"0", "false", "no", "off"},
    )


def end_session(response: Response):
    response.delete_cookie(COOKIE_NAME)


def user_from_request(request: Request) -> Optional[dict]:
    """Return the signed-in user, or None if the cookie is missing, tampered with, expired or no longer allowed."""
    cookie = request.cookies.get(COOKIE_NAME)
    if not cookie:
        return None
    try:
        user = _serializer().loads(cookie, max_age=SESSION_MAX_AGE)
    except BadSignature:  # also covers SignatureExpired
        return None
    if not isinstance(user, dict) or not is_allowed(user.get("email", "")):
        return None
    return user

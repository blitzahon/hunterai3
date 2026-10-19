import os
from typing import Optional
from fastapi import Request, Response
from google.oauth2 import id_token
from google.auth.transport import requests

CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
COOKIE_NAME = "google_user_session"

def client_id() -> str:
    return CLIENT_ID

def enabled() -> bool:
    return bool(CLIENT_ID)

def is_allowed(email: str) -> bool:
    allowed = os.getenv("ALLOWED_EMAILS", "")
    if not allowed:
        return True
    allowed_list = [e.strip().lower() for e in allowed.split(",") if e.strip()]
    return email.lower() in allowed_list

def verify_google_credential(credential: str) -> dict:
    if not CLIENT_ID:
        raise ValueError("GOOGLE_CLIENT_ID is not configured")
    id_info = id_token.verify_oauth2_token(credential, requests.Request(), CLIENT_ID)
    return {
        "email": id_info.get("email"),
        "name": id_info.get("name"),
        "picture": id_info.get("picture"),
    }

def start_session(response: Response, user: dict):
    session_val = f"{user.get('email')}|{user.get('name')}"
    response.set_cookie(
        key=COOKIE_NAME,
        value=session_val,
        httponly=True,
        samesite="lax",
        secure=True,
    )

def end_session(response: Response):
    response.delete_cookie(COOKIE_NAME)

def user_from_request(request: Request) -> Optional[dict]:
    cookie = request.cookies.get(COOKIE_NAME)
    if not cookie or "|" not in cookie:
        return None
    email, name = cookie.split("|", 1)
    return {"email": email, "name": name}
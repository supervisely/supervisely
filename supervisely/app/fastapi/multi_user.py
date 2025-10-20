import hashlib
from contextlib import contextmanager
from typing import Optional, Union

from fastapi import Request

import supervisely.io.env as sly_env
from supervisely.api.module_api import ApiField
from supervisely.app.fastapi.websocket import WebsocketManager
from supervisely.sly_logger import logger


def _parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _user_identity_from_cookie(request: Request) -> Optional[str]:
    cookie_header = request.headers.get("cookie")
    if not cookie_header:
        return None
    return hashlib.sha256(cookie_header.encode("utf-8")).hexdigest()


async def extract_user_id_from_request(request: Request) -> Optional[Union[int, str]]:
    """Extract user ID from various parts of the request."""
    if not sly_env.is_multiuser_mode_enabled():
        return None
    user_id = _parse_int(request.query_params.get("userId"))
    if user_id is None:
        header_user = _parse_int(request.headers.get("x-user-id"))
        if header_user is not None:
            user_id = header_user
    if user_id is None:
        referer = request.headers.get("referer", "")
        if referer:
            from urllib.parse import parse_qs, urlparse

            try:
                parsed_url = urlparse(referer)
                query_params = parse_qs(parsed_url.query)
                referer_user = query_params.get("userId", [None])[0]
                user_id = _parse_int(referer_user)
            except Exception as e:
                logger.error(f"Error parsing userId from referer: {e}")
    if user_id is None and "application/json" in request.headers.get("Content-Type", ""):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        context = payload.get("context") or {}
        user_id = _parse_int(context.get("userId") or context.get(ApiField.USER_ID))
        if user_id is None:
            state_payload = payload.get("state") or {}
            user_id = _parse_int(state_payload.get("userId") or state_payload.get(ApiField.USER_ID))
    if user_id is None:
        user_id = _user_identity_from_cookie(request)
    return user_id


@contextmanager
def session_context(user_id: Optional[Union[int, str]]):
    """
    Context manager to set and reset user context for multiuser applications.
    Call this at the beginning of a request handling to ensure the correct user context is set in environment variables (`supervisely_multiuser_app_user_id` ContextVar).
    """
    if not sly_env.is_multiuser_mode_enabled() or user_id is None:
        yield
        return
    token = sly_env.set_user_for_multiuser_app(user_id)
    try:
        yield
    finally:
        sly_env.reset_user_for_multiuser_app(token)


def remember_cookie(request: Request, user_id: Optional[Union[int, str]]):
    """
    Remember user cookie for the given user ID. This is used to associate WebSocket connections with users in multiuser applications based on cookies.
    Allows WebSocket connections to be correctly routed to the appropriate user.
    """
    if not sly_env.is_multiuser_mode_enabled() or user_id is None:
        return
    cookie_header = request.headers.get("cookie")
    if cookie_header:
        WebsocketManager().remember_user_cookie(cookie_header, user_id)

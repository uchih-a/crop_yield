"""
auth.py — Login / Signup handlers with bcrypt verification.
"""

import logging
import re
from datetime import datetime

import bcrypt
import gradio as gr

from app.database import get_session, User, create_user

logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def authenticate(username: str, password: str):
    """Returns (user_id, username, full_name, is_admin) or (None,)*4."""
    session = get_session()
    try:
        user = session.query(User).filter(
            (User.username == username.strip()) | (User.email == username.strip().lower())
        ).first()
        if user and user.is_active and verify_password(password, user.password_hash):
            user.last_login = datetime.utcnow()
            session.commit()
            return user.id, user.username, user.full_name or user.username, user.is_admin
        return None, None, None, None
    except Exception as e:
        logger.error(f"Auth error: {e}")
        return None, None, None, None
    finally:
        session.close()


def login_handler(username: str, password: str, state: dict):
    if not username or not password:
        return state, gr.update(visible=True), gr.update(visible=False), \
               gr.update(value="⚠️ Please enter your username and password.", visible=True), \
               gr.update(value="")

    uid, uname, fname, is_admin = authenticate(username, password)
    if uid is None:
        return state, gr.update(visible=True), gr.update(visible=False), \
               gr.update(value="❌ Invalid credentials. Please try again.", visible=True), \
               gr.update(value="")

    new_state = {"user_id": uid, "username": uname, "full_name": fname,
                 "is_admin": is_admin, "logged_in": True}
    return (new_state,
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value="", visible=False),
            gr.update(value=""))


def signup_handler(full_name: str, username: str, email: str,
                   password: str, confirm_password: str, state: dict):
    # Validation
    if not all([full_name, username, email, password, confirm_password]):
        return state, gr.update(visible=True), gr.update(visible=False), \
               gr.update(value="⚠️ All fields are required.", visible=True)

    if len(username.strip()) < 3:
        return state, gr.update(visible=True), gr.update(visible=False), \
               gr.update(value="⚠️ Username must be at least 3 characters.", visible=True)

    if not EMAIL_RE.match(email.strip()):
        return state, gr.update(visible=True), gr.update(visible=False), \
               gr.update(value="⚠️ Please enter a valid email address.", visible=True)

    if len(password) < 8:
        return state, gr.update(visible=True), gr.update(visible=False), \
               gr.update(value="⚠️ Password must be at least 8 characters.", visible=True)

    if password != confirm_password:
        return state, gr.update(visible=True), gr.update(visible=False), \
               gr.update(value="⚠️ Passwords do not match.", visible=True)

    user, error = create_user(username, email, password, full_name)
    if error:
        return state, gr.update(visible=True), gr.update(visible=False), \
               gr.update(value=f"❌ {error}", visible=True)

    new_state = {"user_id": user.id, "username": user.username,
                 "full_name": user.full_name or user.username,
                 "is_admin": user.is_admin, "logged_in": True}
    return (new_state,
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value="", visible=False))


def logout_handler(state: dict):
    return ({},
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="", visible=False))

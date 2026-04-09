"""helpers.py — Shared utility functions."""

import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_MONTHS = ["January","February","March","April","May","June",
           "July","August","September","October","November","December"]


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info(".env loaded.")
    else:
        logger.info("No .env — using environment variables.")


def month_name(month_int: int) -> str:
    return _MONTHS[max(0, min(11, int(month_int) - 1))]


def yield_category(val: float) -> str:
    if val < 0.5:  return "Very Low"
    if val < 1.0:  return "Low"
    if val < 1.5:  return "Moderate"
    if val < 2.0:  return "High"
    return "Very High"


def yield_color(val: float) -> str:
    if val < 0.5:  return "#e74c3c"
    if val < 1.0:  return "#e67e22"
    if val < 1.5:  return "#f1c40f"
    if val < 2.0:  return "#2ecc71"
    return "#27ae60"

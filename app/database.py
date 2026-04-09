"""
database.py — MySQL connection, ORM models, and CRUD helpers.
Uses SQLAlchemy 2.x with connection pooling and graceful SQLite fallback.
"""

import os
import logging
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Text, Boolean, ForeignKey, event
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

Base = declarative_base()


# ─────────────────────────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String(80), unique=True, nullable=False, index=True)
    email         = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name     = Column(String(150), nullable=True)
    is_admin      = Column(Boolean, default=False)
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    last_login    = Column(DateTime, nullable=True)

    predictions = relationship("Prediction", back_populates="user",
                               cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    region          = Column(String(100))
    crop            = Column(String(100))
    soil_texture    = Column(String(100))
    month           = Column(Integer)
    rainfall_mm     = Column(Float)
    temperature_c   = Column(Float)
    humidity_pct    = Column(Float)
    soil_ph         = Column(Float)
    soil_sat_pct    = Column(Float)
    land_size_acres = Column(Float)
    predicted_yield = Column(Float)
    yield_category  = Column(String(50))
    created_at      = Column(DateTime, default=datetime.utcnow)
    notes           = Column(Text, nullable=True)

    user = relationship("User", back_populates="predictions")


class CropRecord(Base):
    """Historical dataset rows loaded from CSV."""
    __tablename__ = "crop_records"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    month_year           = Column(String(20), nullable=False, index=True)
    region               = Column(String(100), index=True)
    crop                 = Column(String(100), index=True)
    soil_texture         = Column(String(100))
    rainfall_mm          = Column(Float)
    temperature_c        = Column(Float)
    humidity_pct         = Column(Float)
    soil_ph              = Column(Float)
    soil_saturation_pct  = Column(Float)
    land_size_acres      = Column(Float)
    past_yield_tons_acre = Column(Float)


# ─────────────────────────────────────────────────────────────
# Engine & Session factory
# ─────────────────────────────────────────────────────────────

_engine  = None
_Session = None


def _build_mysql_url() -> str:
    host     = os.environ["DB_HOST"]
    port     = os.getenv("DB_PORT", "3306")
    user     = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    dbname   = os.environ["DB_NAME"]
    return (
        f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}"
        "?charset=utf8mb4"
    )


def init_db(echo: bool = False) -> bool:
    """
    Initialise DB: create tables, seed admin.
    Returns True if MySQL connected, False if fell back to SQLite.
    """
    global _engine, _Session
    try:
        url = _build_mysql_url()
        _engine = create_engine(
            url,
            echo=echo,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=5,
            max_overflow=10,
            poolclass=QueuePool,
        )
        with _engine.connect():
            pass  # probe
        Base.metadata.create_all(_engine)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
        _seed_admin()
        logger.info("✅ MySQL connected and initialised.")
        return True
    except (OperationalError, KeyError) as exc:
        logger.warning(f"MySQL unavailable — SQLite fallback: {exc}")
        _use_sqlite_fallback(echo)
        return False


def _use_sqlite_fallback(echo: bool = False):
    global _engine, _Session
    _engine = create_engine(
        "sqlite:///crop_yield.db",
        echo=echo,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    _seed_admin()
    logger.info("SQLite fallback active.")


def get_session():
    if _Session is None:
        raise RuntimeError("Call init_db() before get_session().")
    return _Session()


def _seed_admin():
    """Create admin user from env if not present."""
    import bcrypt
    session = _Session()
    try:
        admin_username = os.getenv("ADMIN_USERNAME", "admin")
        admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
        admin_email    = os.getenv("ADMIN_EMAIL", "admin@cropyield.ke")

        if not session.query(User).filter_by(username=admin_username).first():
            hashed = bcrypt.hashpw(admin_password.encode(), bcrypt.gensalt()).decode()
            session.add(User(
                username=admin_username,
                email=admin_email,
                password_hash=hashed,
                full_name="System Administrator",
                is_admin=True,
            ))
            session.commit()
            logger.info(f"Admin user '{admin_username}' seeded.")
    except Exception as e:
        session.rollback()
        logger.error(f"Seed error: {e}")
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────
# CRUD helpers
# ─────────────────────────────────────────────────────────────

def create_user(username: str, email: str, password: str, full_name: str = "") -> tuple:
    """
    Register a new user.
    Returns (User, None) on success or (None, error_message) on failure.
    """
    import bcrypt
    session = get_session()
    try:
        if session.query(User).filter_by(username=username.strip()).first():
            return None, "Username already taken."
        if session.query(User).filter_by(email=email.strip().lower()).first():
            return None, "Email already registered."
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        user = User(
            username=username.strip(),
            email=email.strip().lower(),
            password_hash=hashed,
            full_name=full_name.strip(),
        )
        session.add(user)
        session.commit()
        return user, None
    except Exception as e:
        session.rollback()
        logger.error(f"create_user error: {e}")
        return None, "Registration failed. Please try again."
    finally:
        session.close()


def save_prediction(user_id: int, inputs: dict, predicted_yield: float, category: str) -> int | None:
    from utils.helpers import yield_category
    session = get_session()
    try:
        pred = Prediction(
            user_id=user_id,
            region=inputs.get("region"),
            crop=inputs.get("crop"),
            soil_texture=inputs.get("soil_texture"),
            month=inputs.get("month"),
            rainfall_mm=inputs.get("rainfall_mm"),
            temperature_c=inputs.get("temperature_c"),
            humidity_pct=inputs.get("humidity_pct"),
            soil_ph=inputs.get("soil_ph"),
            soil_sat_pct=inputs.get("soil_sat_pct"),
            land_size_acres=inputs.get("land_size_acres"),
            predicted_yield=predicted_yield,
            yield_category=category,
        )
        session.add(pred)
        session.commit()
        return pred.id
    except Exception as e:
        session.rollback()
        logger.error(f"save_prediction error: {e}")
        return None
    finally:
        session.close()


def get_predictions_for_user(user_id: int, limit: int = 100):
    session = get_session()
    try:
        return (
            session.query(Prediction)
            .filter_by(user_id=user_id)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()


def get_all_predictions(limit: int = 500):
    session = get_session()
    try:
        return (
            session.query(Prediction)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()


def bulk_insert_records(records: list):
    """Insert CropRecord dicts in bulk."""
    session = get_session()
    try:
        session.bulk_insert_mappings(CropRecord, records)
        session.commit()
        logger.info(f"Bulk inserted {len(records)} crop records.")
        return True, len(records)
    except Exception as e:
        session.rollback()
        logger.error(f"Bulk insert error: {e}")
        return False, 0
    finally:
        session.close()


def clear_crop_records():
    session = get_session()
    try:
        session.query(CropRecord).delete()
        session.commit()
    finally:
        session.close()


def get_crop_records_df():
    """Return all CropRecord rows as a pandas DataFrame."""
    import pandas as pd
    session = get_session()
    try:
        rows = session.query(CropRecord).all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "Month_Year":           r.month_year,
            "Region":               r.region,
            "Crop":                 r.crop,
            "Soil_Texture":         r.soil_texture,
            "Rainfall_mm":          r.rainfall_mm,
            "Temperature_C":        r.temperature_c,
            "Humidity_pct":         r.humidity_pct,
            "Soil_pH":              r.soil_ph,
            "Soil_Saturation_pct":  r.soil_saturation_pct,
            "Land_Size_acres":      r.land_size_acres,
            "Past_Yield_tons_acre": r.past_yield_tons_acre,
        } for r in rows])
    finally:
        session.close()


def get_db_record_count() -> int:
    session = get_session()
    try:
        return session.query(CropRecord).count()
    finally:
        session.close()

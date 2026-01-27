"""Configuration settings for Smart Money AI Trading System"""

import os
from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "Smart Money AI Trading System"
    DEBUG: bool = True
    SECRET_KEY: str = "your-secret-key-change-in-production"

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/ict_trading"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # API Keys
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    PINECONE_API_KEY: Optional[str] = None

    # Market Data
    POLYGON_API_KEY: Optional[str] = None
    ALPHA_VANTAGE_KEY: Optional[str] = None

    # Notifications
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_WHATSAPP_NUMBER: str = "whatsapp:+14155238886"

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173", "http://127.0.0.1:3000"]

    # Video Processing
    WHISPER_MODEL: str = "large-v3"
    VIDEO_DOWNLOAD_PATH: str = "/tmp/videos"
    TRANSCRIPT_PATH: str = "/tmp/transcripts"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

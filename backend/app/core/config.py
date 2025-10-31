from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env")
    PROJECT_NAME: str = "Waterbody Segmentation API"
    API_V1_STR: str = "/api/v1"

    # CORS settings
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # React frontend default port
        "http://localhost:5173",  # Vite default port
        "http://localhost:8000",  # FastAPI backend default port
    ]

    # Sentinel Hub Configuration
    SENTINEL_HUB_CLIENT_ID: str = os.getenv("SENTINEL_HUB_CLIENT_ID", "")
    SENTINEL_HUB_CLIENT_SECRET: str = os.getenv("SENTINEL_HUB_CLIENT_SECRET", "")

    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(Path(__file__).resolve().parents[3] / "models" / "water_segmentation_model.h5"))

    # File storage
    DATA_DIR: str = os.path.join(str(Path(__file__).resolve().parents[2]), "data")
    INPUT_DIR: str = os.path.join(DATA_DIR, "input")
    OUTPUT_DIR: str = os.path.join(DATA_DIR, "output")

    # API Configuration
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # Pydantic v2 settings via model_config above

settings = Settings()

# Create necessary directories
os.makedirs(settings.INPUT_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

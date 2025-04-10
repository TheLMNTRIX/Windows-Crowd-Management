import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class Settings:
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Tourist Flow & Safety Monitoring System"
    
    # Google API settings
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL: str = "gemini-2.0-flash"
    
    # Video processing settings
    FRAME_SAMPLE_COUNT: int = 5
    UPLOAD_DIR: str = "static/uploads"
    PROCESSED_DIR: str = "static/processed"
    
    # Firebase settings
    FIREBASE_STORAGE_BUCKET: str = "techyothon-456411.firebasestorage.app"
    FIREBASE_SERVICE_ACCOUNT_PATH: str = "firebase-credentials.json"
    
    # Logging settings
    LOG_LEVEL: int = logging.INFO
    LOG_DIR: str = "logs"
    
    # Ensure directories exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

settings = Settings()
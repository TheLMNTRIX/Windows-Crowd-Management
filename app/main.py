import os
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import logging

from app.config import settings
from app.api.routes import router as api_router
from app.utils.helpers import cleanup_old_files
from app.utils.logger import setup_logger

# Set up logger for the main application
logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Setup scheduler for cleanup
scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_old_files, 'interval', hours=24)

@app.on_event("startup")
def startup_event():
    # Ensure directories exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
    
    # Start the scheduler
    scheduler.start()
    
    logger.info(f"Application {settings.PROJECT_NAME} started successfully")
    logger.info(f"API available at {settings.API_V1_STR}")

@app.on_event("shutdown")
def shutdown_event():
    # Shutdown the scheduler
    scheduler.shutdown()
    logger.info("Application shutting down")

@app.get("/")
def root():
    logger.debug("Root endpoint accessed")
    return {
        "name": settings.PROJECT_NAME,
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting development server")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
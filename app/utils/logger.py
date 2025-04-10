import os
import logging
from logging.handlers import RotatingFileHandler
import time
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure the logger
def setup_logger(name):
    """Set up a logger instance with file handling and formatting"""
    logger = logging.getLogger(name)
    
    # Only set up handlers if they don't already exist
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create a file handler for the logs
        log_file = logs_dir / f"{time.strftime('%Y-%m-%d')}.log"
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10485760,  # 10MB max file size
            backupCount=10      # Keep 10 backup copies
        )
        
        # Create a formatter and set it for the handler
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
        )
        file_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(file_handler)
        
        # Also create a stream handler for console output during development
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger
import os
import shutil
from datetime import datetime, timedelta
from app.config import settings

def cleanup_old_files(days=1):
    """Cleanup files older than specified days"""
    now = datetime.now()
    cutoff = now - timedelta(days=days)
    
    # Clean uploads directory
    cleanup_directory(settings.UPLOAD_DIR, cutoff)
    
    # Clean processed directory
    cleanup_directory(settings.PROCESSED_DIR, cutoff)

def cleanup_directory(directory, cutoff):
    """Delete old files from a directory"""
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Skip directories and special files
        if os.path.isdir(filepath) or filename.startswith('.'):
            continue
            
        # Check file modification time
        file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
        if file_modified < cutoff:
            try:
                os.remove(filepath)
                print(f"Deleted old file: {filepath}")
            except Exception as e:
                print(f"Error deleting {filepath}: {e}")
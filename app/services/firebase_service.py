import os
import firebase_admin
from firebase_admin import credentials, storage, firestore
from datetime import datetime
from typing import Dict, Any, List
import asyncio
from app.utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)

class FirebaseService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Initialize Firebase with service account
        try:
            logger.info("Initializing Firebase service")
            cred_path = os.path.join(os.getcwd(), "firebase-credentials.json")
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'techyothon-456411.firebasestorage.app'
            })
            
            # Initialize Firestore and Storage clients
            self.db = firestore.client()
            self.bucket = storage.bucket()
            
            self._initialized = True
            logger.info("Firebase service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {str(e)}")
            raise
            
    def upload_frame_to_storage(self, file_path: str, destination_path: str = None) -> str:
        """
        Upload a frame image to Firebase Storage
        
        Args:
            file_path: Local path to the frame image file
            destination_path: Path in the storage bucket (if None, uses filename)
            
        Returns:
            Public URL of the uploaded file
        """
        try:
            if destination_path is None:
                destination_path = f"frames/{os.path.basename(file_path)}"
                
            blob = self.bucket.blob(destination_path)
            blob.upload_from_filename(file_path)
            
            # Make the file publicly accessible
            blob.make_public()
            
            logger.info(f"Uploaded frame to Firebase Storage: {destination_path}")
            return blob.public_url
        except Exception as e:
            logger.error(f"Error uploading frame to Firebase Storage: {str(e)}")
            raise
    
    async def upload_frames_to_storage_async(self, frame_paths: List[str], video_id: str) -> List[str]:
        """Upload multiple frames to Firebase Storage concurrently"""
        logger.info(f"Starting concurrent upload of {len(frame_paths)} frames")
        
        async def upload_single_frame(frame_path):
            try:
                # Convert relative path to absolute path
                abs_path = os.path.join(os.getcwd(), frame_path.lstrip('/'))
                
                # Run upload in threadpool to prevent blocking
                destination_path = f"videos/{video_id}/frames/{os.path.basename(frame_path)}"
                
                loop = asyncio.get_event_loop()
                cloud_url = await loop.run_in_executor(
                    None,
                    lambda: self.upload_frame_to_storage(abs_path, destination_path)
                )
                
                logger.debug(f"Successfully uploaded frame: {os.path.basename(frame_path)}")
                return cloud_url
            except Exception as e:
                logger.error(f"Error uploading frame {frame_path}: {str(e)}")
                raise
        
        # Create and run upload tasks concurrently
        tasks = [upload_single_frame(path) for path in frame_paths]
        return await asyncio.gather(*tasks)
    
    def store_analysis_data(self, video_id: str, analysis_data: Dict[str, Any], frame_urls: List[str]) -> str:
        """
        Store analysis data in Firestore
        
        Args:
            video_id: Unique ID for the video analysis
            analysis_data: Analysis results and metadata
            frame_urls: List of URLs to frames in Firebase Storage
            
        Returns:
            Document ID in Firestore
        """
        try:
            # Create document with analysis data
            doc_ref = self.db.collection('video-analysis').document(video_id)
            
            # Ensure frame_urls are in the data if they aren't already
            if 'frame_urls' not in analysis_data:
                analysis_data['frame_urls'] = frame_urls
                
            # Add server timestamp
            analysis_data['created_at'] = firestore.SERVER_TIMESTAMP
            
            # Save to Firestore
            doc_ref.set(analysis_data)
            
            logger.info(f"Stored analysis data in Firestore with ID: {video_id}")
            return video_id
        except Exception as e:
            logger.error(f"Error storing analysis data in Firestore: {str(e)}")
            raise
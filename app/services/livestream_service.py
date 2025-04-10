import cv2
import time
import threading
import uuid
import numpy as np
from datetime import datetime
import asyncio
from typing import List, Dict, Any, Optional
import queue
from PIL import Image
from firebase_admin import firestore
import random  # Add this import at the top if not already present

from app.services.video_service import VideoProcessor
from app.services.ai_service import AIService
from app.services.firebase_service import FirebaseService
from app.utils.logger import setup_logger
from app.config import settings

# Set up logger for this module
logger = setup_logger(__name__)

class LivestreamAnalyzer:
    def __init__(self):
        """Initialize the livestream analyzer"""
        self.cap = None
        self.is_running = False
        self.capture_thread = None
        self.analysis_thread = None
        self.frame_buffer = queue.Queue()
        self.current_frames = []
        self.current_timestamps = []
        self.start_time = None
        self.session_id = None
        self.chunk_counter = 0
        
        # Initialize services
        self.ai_service = AIService()
        self.firebase_service = FirebaseService()
        
        # Frame processing settings
        self.analysis_interval = 30  # seconds
        self.frame_sample_rate = 5   # frames per analysis
        self.last_analysis_time = 0
        
        logger.info("LivestreamAnalyzer initialized")

    def start_stream(self, camera_id: int = 0, location_context: Dict[str, Any] = None) -> str:
        """
        Start the livestream analysis
        
        Args:
            camera_id: ID of the camera to use
            location_context: Location data for analysis context
            
        Returns:
            session_id: Unique ID for this livestream session
        """
        if self.is_running:
            logger.warning("Livestream is already running")
            return self.session_id
            
        # Initialize session with a random 6-digit number instead of UUID
        self.session_id = str(random.randint(100000, 999999))
        self.start_time = time.time()
        self.chunk_counter = 0
        self.location_context = location_context
        
        # Create an initial document in Firestore
        self.firebase_service.db.collection('livestream-analysis').document(self.session_id).set({
            'session_id': self.session_id,
            'started_at': datetime.now().isoformat(),
            'camera_id': camera_id,
            'status': 'active',
            'location': location_context,
            'chunks': [],
            'latest_analysis': {},
        })
        
        logger.info(f"Starting livestream with session ID: {self.session_id}")
        
        # Open camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera with ID {camera_id}")
            raise ValueError(f"Failed to open camera with ID {camera_id}")
        
        # Start capture thread
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self._analyze_chunks)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        return self.session_id
    
    def stop_stream(self) -> Dict[str, Any]:
        """
        Stop the livestream and return the final results
        
        Returns:
            Summary of the livestream session
        """
        if not self.is_running:
            logger.warning("No livestream is running")
            return {'error': 'No active livestream'}
        
        # Stop the threads
        self.is_running = False
        
        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2.0)
        
        # Release camera
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        # Update Firebase document
        if self.session_id:
            self.firebase_service.db.collection('livestream-analysis').document(self.session_id).update({
                'ended_at': datetime.now().isoformat(),
                'status': 'completed',
                'total_duration': time.time() - self.start_time,
                'chunk_count': self.chunk_counter
            })
            
            # Get final document
            doc = self.firebase_service.db.collection('livestream-analysis').document(self.session_id).get()
            
            logger.info(f"Livestream session {self.session_id} completed with {self.chunk_counter} analysis chunks")
            return doc.to_dict()
        
        return {'error': 'Session ID not found'}
    
    def get_session_info(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about the current or a specific session
        
        Args:
            session_id: Optional ID of a session to look up
            
        Returns:
            Session information
        """
        id_to_check = session_id or self.session_id
        
        if not id_to_check:
            return {'error': 'No session ID provided or active'}
        
        doc = self.firebase_service.db.collection('livestream-analysis').document(id_to_check).get()
        
        if not doc.exists:
            return {'error': 'Session not found'}
            
        return doc.to_dict()

    def _capture_frames(self):
        """Continuously capture frames from the camera in a separate thread"""
        logger.info("Frame capture thread started")
        
        frame_count = 0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:  # If FPS isn't available, estimate it
            fps = 30
        
        # Current analysis results to display
        current_analysis = {}
        
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                logger.error("Failed to capture frame from camera")
                self.is_running = False
                break
            
            # Calculate timestamp
            current_time = time.time() - self.start_time
            
            # Store frames for analysis at reduced rate
            frame_count += 1
            if frame_count % int(fps / 5) == 0:  # Store ~5 frames per second for analysis
                # Convert BGR to RGB for analysis
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frames.append(rgb_frame)
                self.current_timestamps.append(current_time)
            
            # Display the frame with a timestamp
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)
            time_text = f"{minutes:02d}:{seconds:02d}"
            
            # Get the latest analysis data from Firestore every few seconds
            if frame_count % int(fps * 3) == 0:  # Update every 3 seconds
                try:
                    doc = self.firebase_service.db.collection('livestream-analysis').document(self.session_id).get()
                    if doc.exists:
                        data = doc.to_dict()
                        current_analysis = data.get('latest_analysis', {})
                except Exception as e:
                    logger.error(f"Error fetching latest analysis: {str(e)}")
            
            # Add colored border based on crowd level
            border_width = 10
            if current_analysis:
                # Define color based on crowd level (BGR format)
                crowd_level = int(current_analysis.get('crowd_level', 0))
                intervention_required = current_analysis.get('police_intervention_required', False)
                
                if intervention_required:
                    border_color = (0, 0, 255)  # Red for intervention required
                elif crowd_level >= 7:
                    border_color = (0, 140, 255)  # Orange for high crowd
                elif crowd_level >= 4:
                    border_color = (0, 255, 255)  # Yellow for medium crowd
                else:
                    border_color = (0, 255, 0)    # Green for low crowd
                    
                # Add colored border
                frame = cv2.copyMakeBorder(
                    frame, 
                    border_width, border_width, border_width, border_width,
                    cv2.BORDER_CONSTANT, 
                    value=border_color
                )
            
            # Add session ID and time to the frame
            cv2.putText(
                frame, 
                f"Session: {self.session_id} | Time: {time_text}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255), 
                2
            )
            
            # Add analysis info if available
            if current_analysis:
                crowd_level = current_analysis.get('crowd_level', 'N/A')
                crowd_count = current_analysis.get('crowd_count', 'N/A')
                is_peak = 'Yes' if current_analysis.get('is_peak_hour', False) else 'No'
                
                y_pos = 70
                cv2.putText(
                    frame,
                    f"Crowd Level: {crowd_level}/10 | Count: {crowd_count} | Peak Hour: {is_peak}",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Add police intervention status
                if current_analysis.get('police_intervention_required', False):
                    y_pos += 30
                    cv2.putText(
                        frame,
                        "POLICE INTERVENTION REQUIRED",
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),  # Red color for warning
                        2
                    )
            
            # Show the frame
            cv2.imshow('Livestream Analysis', frame)
            
            # Check for 'q' key to stop the stream
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                break
    
    def _analyze_chunks(self):
        """Analyze 30-second chunks of the livestream in a separate thread"""
        logger.info("Analysis thread started")
        
        while self.is_running:
            current_time = time.time() - self.start_time
            
            # Check if it's time for an analysis
            if current_time - self.last_analysis_time >= self.analysis_interval and len(self.current_frames) > 0:
                logger.info(f"Starting analysis of 30-second chunk {self.chunk_counter + 1}")
                
                # Copy current frames to analyze
                frames_to_analyze = self.current_frames.copy()
                timestamps_to_analyze = self.current_timestamps.copy()
                
                # Clear the buffers
                self.current_frames = []
                self.current_timestamps = []
                
                # Update last analysis time
                self.last_analysis_time = current_time
                self.chunk_counter += 1
                
                # Select a subset of frames if there are too many
                if len(frames_to_analyze) > self.frame_sample_rate:
                    indices = np.linspace(0, len(frames_to_analyze) - 1, self.frame_sample_rate, dtype=int)
                    frames_to_analyze = [frames_to_analyze[i] for i in indices]
                    timestamps_to_analyze = [timestamps_to_analyze[i] for i in indices]
                
                # Perform analysis in a separate thread to not block
                analysis_thread = threading.Thread(
                    target=self._process_and_store_analysis,
                    args=(frames_to_analyze, timestamps_to_analyze, self.chunk_counter)
                )
                analysis_thread.daemon = True
                analysis_thread.start()
            
            # Sleep to avoid high CPU usage
            time.sleep(1)
    
    def _process_and_store_analysis(self, frames, timestamps, chunk_number):
        """Process frames and store the analysis results"""
        try:
            # Get chunk time range
            start_time = timestamps[0] if timestamps else 0
            end_time = timestamps[-1] if timestamps else 0
            
            # Generate unique ID for this chunk
            chunk_id = f"{self.session_id}_chunk_{chunk_number}"
            
            # Save frames as images
            frame_paths = []
            for i, frame in enumerate(frames):
                # Convert to PIL image
                pil_image = np.asarray(frame)
                
                # Create filename
                filename = f"{chunk_id}_frame_{i}.jpg"
                file_path = f"{settings.PROCESSED_DIR}/{filename}"
                
                # Save image
                cv2.imwrite(file_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                # Add to list
                frame_paths.append(f"/static/processed/{filename}")
            
            # Analyze the frames
            pil_frames = [Image.fromarray(frame) for frame in frames]
            
            # Run analysis asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            analysis_result = loop.run_until_complete(
                self.ai_service.analyze_frames_parallel(pil_frames, timestamps, self.location_context)
            )
            loop.close()
            
            # Upload frames to Firebase Storage
            storage_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(storage_loop)
            cloud_urls = storage_loop.run_until_complete(
                self.firebase_service.upload_frames_to_storage_async(frame_paths, chunk_id)
            )
            storage_loop.close()
            
            # Store analysis in Firestore
            chunk_data = {
                'chunk_id': chunk_id,
                'chunk_number': chunk_number,
                'time_range': {
                    'start': start_time,
                    'end': end_time
                },
                'frame_urls': cloud_urls,
                'analysis': analysis_result,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update the session document
            self.firebase_service.db.collection('livestream-analysis').document(self.session_id).update({
                'chunks': firestore.ArrayUnion([chunk_data]),
                'latest_analysis': analysis_result,
                'last_updated': datetime.now().isoformat()
            })
            
            logger.info(f"Completed analysis of chunk {chunk_number}")
            
        except Exception as e:
            logger.error(f"Error processing analysis for chunk {chunk_number}: {str(e)}")
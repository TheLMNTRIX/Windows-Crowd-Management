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
        self.display_thread = None  # Add display thread
        self.frame_buffer = queue.Queue()
        self.current_frames = []
        self.current_timestamps = []
        self.start_time = None
        self.session_id = None
        self.chunk_counter = 0
        self.camera_id = None
        
        # For continuous display
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.latest_analysis = None
        
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
        self.camera_id = camera_id
        self.latest_analysis = None
        
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
        
        # Start threads
        self.is_running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start display thread
        self.display_thread = threading.Thread(target=self._display_live_feed)
        self.display_thread.daemon = True
        self.display_thread.start()
        
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
        
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2.0)
        
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
        """Capture frames continuously and process in 30-second chunks"""
        logger.info("Frame capture thread started")
        
        chunk_frames = []
        chunk_timestamps = []
        chunk_start_time = time.time()
        current_chunk_number = self.chunk_counter + 1
        
        logger.info(f"Starting to capture chunk {current_chunk_number}")
        
        while self.is_running:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Update current frame for display thread
            with self.frame_lock:
                self.current_frame = frame
            
            # Calculate timestamp relative to stream start
            current_time = time.time() - self.start_time
            
            # Store the frame for chunk processing (sample every few frames for efficiency)
            if len(chunk_frames) % 6 == 0:  # Assume 30fps, sample at ~5fps
                chunk_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                chunk_timestamps.append(current_time)
            
            # Check if the current chunk duration is complete
            if (time.time() - chunk_start_time) >= self.analysis_interval:
                # If we've collected frames
                if chunk_frames:
                    logger.info(f"Completed capturing chunk {current_chunk_number} - {len(chunk_frames)} frames")
                    
                    # Increment chunk counter
                    self.chunk_counter = current_chunk_number
                    
                    # Select a subset of frames if there are too many
                    if len(chunk_frames) > self.frame_sample_rate:
                        indices = np.linspace(0, len(chunk_frames) - 1, self.frame_sample_rate, dtype=int)
                        chunk_frames_sample = [chunk_frames[i] for i in indices]
                        chunk_timestamps_sample = [chunk_timestamps[i] for i in indices]
                    else:
                        chunk_frames_sample = chunk_frames
                        chunk_timestamps_sample = chunk_timestamps
                    
                    # Process this chunk in a separate thread
                    processing_thread = threading.Thread(
                        target=self._process_and_analyze_chunk,
                        args=(chunk_frames.copy(), chunk_frames_sample.copy(), chunk_timestamps_sample.copy(), current_chunk_number)
                    )
                    processing_thread.daemon = True
                    processing_thread.start()
                
                # Start a new chunk
                chunk_frames = []
                chunk_timestamps = []
                chunk_start_time = time.time()
                current_chunk_number = self.chunk_counter + 1
                logger.info(f"Starting to capture chunk {current_chunk_number}")
            
            # Brief pause to reduce CPU usage
            time.sleep(0.01)

    def _process_and_analyze_chunk(self, all_frames, sample_frames, timestamps, chunk_number):
        """Process, analyze and store chunk data without blocking live display"""
        try:
            logger.info(f"Processing chunk {chunk_number} with {len(sample_frames)} sample frames")
            
            # Generate unique ID for this chunk
            chunk_id = f"{self.session_id}_chunk_{chunk_number}"
            
            # Process the sample frames with crowd counter first
            crowd_results = []
            for frame in sample_frames:
                # Process with crowd counter
                crowd_data = self.ai_service.crowd_counter.count_crowd(frame)
                crowd_results.append(crowd_data)
            
            # Aggregate crowd counting results
            avg_count = sum(int(result["crowd_count"]) for result in crowd_results) / len(crowd_results)
            max_level = max(int(result["crowd_level"]) for result in crowd_results)
            
            crowd_info = {
                "crowd_count": str(int(round(avg_count))),
                "crowd_level": str(max_level)
            }
            
            # Update the latest analysis for display
            self.latest_analysis = crowd_info
            
            # Start deeper analysis and storage in background
            analysis_thread = threading.Thread(
                target=self._save_analysis_data,
                args=(sample_frames, timestamps, chunk_number, chunk_id, crowd_info)
            )
            analysis_thread.daemon = True
            analysis_thread.start()
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_number}: {str(e)}")

    def _save_analysis_data(self, frames, timestamps, chunk_number, chunk_id, crowd_info):
        """Save frames and perform deeper analysis with Gemini"""
        try:
            # Save frames as images
            frame_paths = []
            for i, frame in enumerate(frames):
                # Create filename
                filename = f"{chunk_id}_frame_{i}.jpg"
                file_path = f"{settings.PROCESSED_DIR}/{filename}"
                
                # Save image
                cv2.imwrite(file_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                # Add to list
                frame_paths.append(f"/static/processed/{filename}")
            
            # Analyze the frames with Gemini AI
            pil_frames = [Image.fromarray(frame) for frame in frames]
            
            # Run analysis asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            analysis_result = loop.run_until_complete(
                self.ai_service.analyze_frames_parallel(pil_frames, timestamps, self.location_context)
            )
            loop.close()
            
            # Make sure our crowd data is preserved
            analysis_result["crowd_count"] = crowd_info["crowd_count"]
            analysis_result["crowd_level"] = crowd_info["crowd_level"]
            
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
                    'start': timestamps[0] if timestamps else 0,
                    'end': timestamps[-1] if timestamps else 0
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
            
            logger.info(f"Completed full analysis of chunk {chunk_number}")
            
        except Exception as e:
            logger.error(f"Error saving analysis data for chunk {chunk_number}: {str(e)}")

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

    def _display_live_feed(self):
        """Display the live camera feed with the latest analysis overlay"""
        logger.info("Display thread started")
        
        while self.is_running:
            # Get the current frame
            with self.frame_lock:
                if self.current_frame is None:
                    time.sleep(0.03)
                    continue
                frame = self.current_frame.copy()
            
            # Add analysis overlay if available
            if self.latest_analysis:
                crowd_level = int(self.latest_analysis["crowd_level"])
                count = self.latest_analysis["crowd_count"]
                
                # Add colored border based on crowd level
                if crowd_level >= 7:
                    border_color = (0, 0, 255)  # Red for high crowd (BGR)
                elif crowd_level >= 4:
                    border_color = (0, 165, 255)  # Orange for medium crowd
                else:
                    border_color = (0, 255, 0)  # Green for low crowd
                
                # Add border
                frame = cv2.copyMakeBorder(
                    frame, 30, 30, 30, 30,
                    cv2.BORDER_CONSTANT,
                    value=border_color
                )
                
                # Add text with crowd info
                cv2.putText(
                    frame,
                    f"LIVE | Latest Analysis (Chunk {self.chunk_counter}) | Crowd: {count} | Level: {crowd_level}/10",
                    (40, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            
            # Show the frame
            cv2.imshow("Crowd Analysis (Live)", frame)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                break
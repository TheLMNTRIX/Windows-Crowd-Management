import os
import cv2
import uuid
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
from typing import List, Tuple, Dict
import json

from app.config import settings

class VideoProcessor:
    @staticmethod
    def save_upload_file(file, filename: str = None) -> str:
        """Save uploaded file to disk and return the path"""
        if not filename:
            filename = f"{uuid.uuid4()}.mp4"
        
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        
        # Write file to disk
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        
        return file_path
    
    @staticmethod
    def extract_key_frames(video_path: str, num_frames: int = 5, resize_factor: float = 0.5) -> Tuple[List[np.ndarray], List[float], float]:
        """Extract evenly distributed frames from video with resizing for faster processing"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        # Calculate the frames to extract (evenly distributed)
        frame_indices = [int((i * total_frames) / num_frames) for i in range(num_frames)]
        
        frames = []
        timestamps = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize frame for faster processing
                if resize_factor != 1.0:
                    height, width = frame.shape[:2]
                    new_height, new_width = int(height * resize_factor), int(width * resize_factor)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                # Calculate timestamp in seconds
                timestamp = idx / fps
                timestamps.append(timestamp)
        
        cap.release()
        
        return frames, timestamps, duration
    
    @staticmethod
    def extract_key_frames_smart(video_path: str, num_frames: int = 5) -> Tuple[List[np.ndarray], List[float], float]:
        """Extract key frames based on content changes rather than just equal intervals"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        # Initialize variables
        frames = []
        timestamps = []
        prev_frame = None
        frame_diff_scores = []
        
        # Sample frames for difference calculation
        sample_rate = max(1, total_frames // 30)  # Examine ~30 frames for differences
        
        # Calculate differences between frames
        for i in range(0, total_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale for faster comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate absolute difference between frames
                diff = cv2.absdiff(gray, prev_frame)
                diff_score = np.sum(diff)
                frame_diff_scores.append((i, diff_score))
                
            prev_frame = gray
        
        # Select frames with highest differences (most content changes)
        # Always include first and last frame
        frame_diff_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [0, total_frames-1]  # First and last frames
        selected_indices.extend([idx for idx, _ in frame_diff_scores[:num_frames-2]])
        selected_indices = sorted(list(set(selected_indices)))[:num_frames]
        
        # Extract the selected frames
        for idx in selected_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                timestamp = idx / fps
                timestamps.append(timestamp)
        
        cap.release()
        return frames, timestamps, duration
    
    @staticmethod
    def save_frames_as_images(frames: List[np.ndarray], video_id: str) -> List[str]:
        """Save frames as image files and return their paths"""
        image_paths = []
        
        for i, frame in enumerate(frames):
            # Convert to PIL image
            pil_image = Image.fromarray(frame)
            
            # Create filename
            filename = f"{video_id}_frame_{i}.jpg"
            file_path = os.path.join(settings.PROCESSED_DIR, filename)
            
            # Save image
            pil_image.save(file_path)
            
            # Add to list
            image_paths.append(f"/static/processed/{filename}")
        
        return image_paths
    
    @staticmethod
    def create_analysis_video(video_path: str, analysis_result: str, video_id: str) -> str:
        """Create annotated video with analysis results"""
        # Load the video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output filename
        output_filename = f"{video_id}_analyzed.mp4"
        output_path = os.path.join(settings.PROCESSED_DIR, output_filename)
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width + 60, height + 60))
        
        # Get alert level
        try:
            analysis_data = json.loads(analysis_result) if isinstance(analysis_result, str) else analysis_result
            alert_level = int(analysis_data.get("alert_level", 0))
            
            # Define color based on alert level (BGR format)
            if alert_level == 2:
                border_color = (0, 0, 255)  # Red for high alert
                label = "HIGH ALERT"
            elif alert_level == 1:
                border_color = (0, 165, 255)  # Orange for medium alert
                label = "MEDIUM ALERT"
            else:
                border_color = (0, 255, 0)  # Green for low alert
                label = "LOW ALERT"
        except:
            border_color = (255, 255, 255)  # White if there's an error
            label = "ANALYSIS ERROR"
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Add colored border
            frame = cv2.copyMakeBorder(frame, 30, 30, 30, 30, 
                                       cv2.BORDER_CONSTANT, value=border_color)
            
            # Add label
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add timestamp
            timestamp = frame_count / fps
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            cv2.putText(frame, time_str, (width - 100, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            out.write(frame)
            frame_count += 1
            
        cap.release()
        out.release()
        
        return f"/static/processed/{output_filename}"
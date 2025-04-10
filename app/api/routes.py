import os
import uuid
import random
from datetime import datetime, time
import pytz
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from PIL import Image
from typing import List, Dict, Any
import logging
from firebase_admin import firestore
from fastapi import Query
from typing import List, Dict, Any
from app.services.video_service import VideoProcessor
from app.services.ai_service import AIService
from app.services.firebase_service import FirebaseService
from app.models.schemas import VideoAnalysisResponse, AnalysisResponse
from app.config import settings
import asyncio

router = APIRouter()
ai_service = AIService()
firebase_service = FirebaseService()  # Initialize Firebase service

# Define Kolkata timezone
kolkata_tz = pytz.timezone('Asia/Kolkata')

# Define peak and non-peak hours (in Kolkata IST)
PEAK_HOURS = [
    "08:30:00",  # Morning rush
    "12:30:00",  # Lunch hour
    "15:00:00",  # Afternoon tourist peak
    "17:30:00",  # Evening rush
    "19:00:00",  # Dinner time
]

NON_PEAK_HOURS = [
    "06:00:00",  # Early morning
    "14:00:00",  # Mid-afternoon
    "22:00:00",  # Late night
]

logger = logging.getLogger(__name__)

@router.post("/analyze-video/", response_model=VideoAnalysisResponse)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    latitude: float = Query(..., description="Latitude of the video location"),
    longitude: float = Query(..., description="Longitude of the video location")
):
    """Analyze a video file for crowd flow and safety monitoring with location data"""
    # Check if file is a video
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Generate a unique ID for this video analysis
        video_id = str(uuid.uuid4())
        
        # Select a timestamp from the predefined options
        # 5/8 chance of selecting a peak hour
        is_peak = random.random() < 0.625  # 5/8 = 0.625
        time_str = random.choice(PEAK_HOURS if is_peak else NON_PEAK_HOURS)
        
        # Create current date with selected time in Kolkata timezone
        now = datetime.now(kolkata_tz)
        time_parts = [int(x) for x in time_str.split(':')]
        location_time = now.replace(hour=time_parts[0], minute=time_parts[1], second=time_parts[2])
        location_timestamp = location_time.isoformat()
        
        # Save the uploaded file
        video_path = VideoProcessor.save_upload_file(file, f"{video_id}.mp4")
        
        # Extract frames with resizing for faster processing
        frames, timestamps, duration = VideoProcessor.extract_key_frames(
            video_path, 
            num_frames=settings.FRAME_SAMPLE_COUNT,
            resize_factor=0.5  # Resize to 50% for faster processing
        )
        
        # Save frames as images
        frame_paths = VideoProcessor.save_frames_as_images(frames, video_id)
        
        # Convert numpy arrays to PIL images for AI service
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Create location context for AI
        location_context = {
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": location_timestamp,
            "local_time": time_str,
            "is_peak_hour_preselected": is_peak
        }
        
        # Run both operations concurrently
        logger.info("Starting concurrent frame uploads and AI analysis")
        upload_task = firebase_service.upload_frames_to_storage_async(frame_paths, video_id)
        analysis_task = ai_service.analyze_frames_parallel(pil_frames, timestamps, location_context)
        
        # Wait for both tasks to complete
        cloud_frame_urls, analysis_result = await asyncio.gather(upload_task, analysis_task)
        logger.info("Concurrent operations completed successfully")
        
        # Create the analyzed video in the background
        background_tasks.add_task(
            VideoProcessor.create_analysis_video,
            video_path, 
            analysis_result, 
            video_id
        )
        
        # Convert raw analysis to AnalysisResponse model
        analysis = AnalysisResponse(
            crowd_present=str(analysis_result.get("crowd_present", 0)),
            crowd_level=str(analysis_result.get("crowd_level", 0)),
            crowd_count=str(analysis_result.get("crowd_count", 0)),
            is_peak_hour=str(analysis_result.get("is_peak_hour", 0)),
            police_intervention_required=str(analysis_result.get("police_intervention_required", 0)),
            police_intervention_suggestions=analysis_result.get("police_intervention_suggestions", [])
        )
        
        # Create response
        response = VideoAnalysisResponse(
            analysis=analysis,
            original_video_url=f"/static/uploads/{video_id}.mp4",
            annotated_video_url=f"/static/uploads/{video_id}_annotated.mp4",
            extracted_frames_urls=frame_paths,
            video_duration=duration,
            timestamp=datetime.now().isoformat(),
            location_latitude=latitude,
            location_longitude=longitude,
            location_timestamp=location_timestamp
        )
        
        # Store analysis data in Firestore
        firebase_service.store_analysis_data(
            video_id=video_id,
            analysis_data={
                "analysis": {
                    "crowd_present": analysis.crowd_present,
                    "crowd_level": analysis.crowd_level,
                    "crowd_count": analysis.crowd_count,
                    "is_peak_hour": analysis.is_peak_hour,
                    "police_intervention_required": analysis.police_intervention_required,
                    "police_intervention_suggestions": analysis.police_intervention_suggestions,
                },
                "location": {
                    "latitude": latitude,
                    "longitude": longitude,
                    "timestamp": location_timestamp
                },
                "video_duration": duration,
                "timestamp": datetime.now().isoformat()
            },
            frame_urls=cloud_frame_urls
        )
        
        # Remove original file in background task
        background_tasks.add_task(os.remove, video_path)
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@router.get("/video-analysis/", response_model=List[Dict[str, Any]])
async def get_video_analyses(limit: int = Query(10, description="Number of records to return")):
    """Retrieve video analysis results from Firestore"""
    try:
        # Initialize Firebase service
        fb_service = FirebaseService()
        
        # Get documents from Firestore collection
        analyses = []
        query = fb_service.db.collection('video-analysis').order_by('created_at', direction=firestore.Query.DESCENDING).limit(limit)
        docs = query.stream()
        
        # Convert documents to dictionaries
        for doc in docs:
            data = doc.to_dict()
            # Add document ID if it's not already included
            if 'video_id' not in data:
                data['video_id'] = doc.id
            analyses.append(data)
        
        logger.info(f"Retrieved {len(analyses)} video analysis records from Firestore")
        return analyses
    
    except Exception as e:
        logger.error(f"Error retrieving video analyses from Firestore: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving video analyses: {str(e)}")
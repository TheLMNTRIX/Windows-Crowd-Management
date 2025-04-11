import os
import uuid
import random
from datetime import datetime, time
import pytz
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
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
from app.services.livestream_service import LivestreamAnalyzer
from app.models.schemas import VideoAnalysisResponse, AnalysisResponse
from app.config import settings
import asyncio

router = APIRouter()
ai_service = AIService()
firebase_service = FirebaseService()  # Initialize Firebase service
livestream_analyzer = LivestreamAnalyzer()  # Create a livestream analyzer instance

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
        
        # Use stream-based video analysis
        frames, crowd_results, duration = VideoProcessor.analyze_video_with_stream(
            video_path, 
            ai_service,
            max_frames=settings.FRAME_SAMPLE_COUNT
        )
        
        # Calculate timestamps for frames
        timestamps = [i * (duration / len(frames)) for i in range(len(frames))]
        
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
        
        # Get crowd info from processed results
        crowd_info = {
            "crowd_count": str(max(int(r["crowd_count"]) for r in crowd_results)),
            "crowd_level": str(max(int(r["crowd_level"]) for r in crowd_results))
        }
        
        # Run both operations concurrently
        logger.info("Starting concurrent frame uploads and AI analysis with Gemini")
        upload_task = firebase_service.upload_frames_to_storage_async(frame_paths, video_id)
        
        # Run Gemini analysis with the crowd info from video stream processing
        analysis_task = ai_service.analyze_frames_with_crowd_info(
            pil_frames, 
            timestamps, 
            location_context,
            crowd_info
        )
        
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
        
        # Prepare the analysis data structure that will be both stored and returned
        analysis_data = {
            "video_id": video_id,
            "analysis": {
                "crowd_present": str(analysis_result.get("crowd_present", False)).lower(),
                "crowd_level": analysis_result.get("crowd_level", "0"),
                "crowd_count": analysis_result.get("crowd_count", "0"),
                "is_peak_hour": str(analysis_result.get("is_peak_hour", False)).lower(),
                "police_intervention_required": str(analysis_result.get("police_intervention_required", False)).lower(),
                "police_intervention_suggestions": analysis_result.get("police_intervention_suggestions", [])
            },
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "timestamp": location_timestamp
            },
            "frame_urls": cloud_frame_urls,
            "video_duration": duration,
            "timestamp": datetime.now().isoformat(),
            "original_video_url": f"/static/uploads/{video_id}.mp4",
            "annotated_video_url": f"/static/processed/{video_id}_analyzed.mp4"  # Updated path to match what VideoProcessor.create_analysis_video returns
        }
        
        # Store analysis data in Firestore
        firebase_service.store_analysis_data(
            video_id=video_id,
            analysis_data=analysis_data.copy(),  # Use a copy to avoid modifying the return data
            frame_urls=cloud_frame_urls
        )
        
        # Remove original file in background task
        background_tasks.add_task(os.remove, video_path)
        
        return analysis_data
    
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

@router.post("/livestream/start/", tags=["Livestream"])
async def start_livestream(
    camera_id: int = Query(0, description="ID of the camera to use"),
    latitude: float = Query(None, description="Latitude of the location (optional)"),
    longitude: float = Query(None, description="Longitude of the location (optional)"),
):
    """Start analyzing a livestream from the camera"""
    try:
        # Create location context if coordinates provided
        location_context = None
        if latitude is not None and longitude is not None:
            location_context = {
                "latitude": latitude,
                "longitude": longitude,
                "timestamp": datetime.now(kolkata_tz).isoformat(),
                "local_time": datetime.now(kolkata_tz).strftime("%H:%M:%S")
            }
        
        # Start the livestream
        session_id = livestream_analyzer.start_stream(camera_id, location_context)
        
        return {
            "status": "success",
            "message": "Livestream started",
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start livestream: {str(e)}")

@router.post("/livestream/stop/", tags=["Livestream"])
async def stop_livestream():
    """Stop the current livestream analysis"""
    try:
        result = livestream_analyzer.stop_stream()
        
        return {
            "status": "success",
            "message": "Livestream stopped",
            "summary": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop livestream: {str(e)}")

@router.get("/livestream/status/", tags=["Livestream"])
async def get_livestream_status(session_id: str = Query(None, description="Session ID to check (optional)")):
    """Get the status of the current or a specific livestream session"""
    try:
        info = livestream_analyzer.get_session_info(session_id)
        
        return {
            "status": "success",
            "session_info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get livestream status: {str(e)}")

@router.get("/livestream/analysis/{session_id}", tags=["Livestream"])
async def get_livestream_analysis(
    session_id: str,
    include_frames: bool = Query(False, description="Include frame URLs in response"),
    limit_chunks: int = Query(None, description="Limit number of chunks returned")
):
    """Get detailed analysis results for a livestream session"""
    try:
        # Get the document
        fb_service = FirebaseService()
        doc = fb_service.db.collection('livestream-analysis').document(session_id).get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            
        data = doc.to_dict()
        
        # Process chunks - sort by time and apply limit
        if "chunks" in data:
            # Sort chunks by chunk_number
            data["chunks"] = sorted(data["chunks"], key=lambda x: x.get("chunk_number", 0))
            
            # Apply limit if specified
            if limit_chunks is not None and limit_chunks > 0:
                data["chunks"] = data["chunks"][:limit_chunks]
                
            # Remove frame URLs if not requested (to reduce response size)
            if not include_frames:
                for chunk in data["chunks"]:
                    if "frame_urls" in chunk:
                        chunk["frame_count"] = len(chunk["frame_urls"])
                        del chunk["frame_urls"]
                        
        # Extract time series data for visualization
        time_series = []
        if "chunks" in data:
            for chunk in data["chunks"]:
                if "analysis" in chunk and "time_range" in chunk:
                    time_series.append({
                        "chunk_number": chunk.get("chunk_number"),
                        "time": chunk.get("time_range", {}).get("end"),
                        "crowd_level": chunk.get("analysis", {}).get("crowd_level", 0),
                        "crowd_count": chunk.get("analysis", {}).get("crowd_count", "0"),
                        "police_intervention_required": chunk.get("analysis", {}).get("police_intervention_required", False),
                    })
        
        # Add time series data to response
        data["time_series"] = time_series
        
        return {
            "status": "success",
            "session_id": session_id,
            "analysis_data": data
        }
    except Exception as e:
        logger.error(f"Error retrieving livestream analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving livestream analysis: {str(e)}")
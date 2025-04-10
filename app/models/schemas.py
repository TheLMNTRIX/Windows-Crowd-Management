from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Any

class AnalysisResponse(BaseModel):
    crowd_present: str
    crowd_count: str
    crowd_level: str
    is_peak_hour: str
    police_intervention_required: str
    police_intervention_suggestions: List[str]


class VideoAnalysisResponse(BaseModel):
    analysis: AnalysisResponse
    original_video_url: str
    annotated_video_url: str
    extracted_frames_urls: List[str]
    video_duration: float
    timestamp: str
    location_latitude: float
    location_longitude: float
    location_timestamp: str
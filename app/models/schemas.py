from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Any

class AnalysisResponse(BaseModel):
    crowd_present: str
    crowd_count: str
    crowd_level: str
    is_peak_hour: str
    police_intervention_required: str
    police_intervention_suggestions: List[str]


class LocationData(BaseModel):
    latitude: float
    longitude: float
    timestamp: str

class VideoAnalysisResponse(BaseModel):
    video_id: str
    analysis: AnalysisResponse
    location: LocationData
    frame_urls: List[str]
    video_duration: float
    timestamp: str
    original_video_url: str
    annotated_video_url: str
    created_at: Optional[Any] = None
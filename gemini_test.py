# gemini_test.py - Standalone test app for Gemini API

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import google.generativeai as genai
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# Configure Gemini
genai.configure(api_key=api_key)
MODEL_NAME = "gemini-2.5-pro-exp-03-25"  # You can change this if needed

# Create FastAPI app
app = FastAPI(
    title="Gemini API Test",
    description="Test endpoints for verifying Gemini API integration for crowd analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class TestResponse(BaseModel):
    status: str
    model_used: str
    test_response: str
    api_status: str

class ImageAnalysisResponse(BaseModel):
    status: str
    analysis: Dict[str, Any]

# Test endpoints
@app.get("/test-gemini/", response_model=TestResponse, tags=["Tests"])
async def test_gemini_integration():
    """
    Test if Gemini API integration is working properly.
    
    This endpoint makes a simple request to the Gemini API and returns the response.
    Use this to verify your API key and connectivity.
    """
    try:
        # Create model
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Simple test prompt
        test_prompt = "Hello! This is a test prompt to check if Gemini integration is working. Please respond with a short greeting and confirm you can process tourist crowd analysis."
        
        # Generate content with Gemini
        response = model.generate_content(test_prompt)
        
        # Return successful response
        return TestResponse(
            status="success",
            model_used=MODEL_NAME,
            test_response=response.text,
            api_status="connected"
        )
    
    except Exception as e:
        # Return error information
        return TestResponse(
            status="error",
            model_used=MODEL_NAME,
            test_response=f"Error: {str(e)}",
            api_status="failed"
        )

@app.post("/test-image-analysis/", response_model=ImageAnalysisResponse, tags=["Tests"])
async def test_image_analysis(file: UploadFile = File(...)):
    """
    Test endpoint to analyze a single image for crowd conditions.
    
    Upload an image file with crowds to test if Gemini can analyze it properly.
    This is useful for testing the AI's ability to assess crowd density before
    implementing the full video analysis pipeline.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Create model
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Simplified prompt for single image
        prompt = (
            "You are an expert crowd analysis system specializing in tourist flow and safety monitoring."
            "Your task is to provide a structured and detailed analysis of crowd conditions from the image. Follow these instructions:\n\n"
            
            "**1. Assess Crowd Density and Flow:**\n"
            "- Identify areas of high, medium, and low crowd density.\n"
            "- Analyze movement patterns, flow directions, and bottlenecks.\n"
            "- Detect unusual congestion points or rapid crowd density changes.\n\n"
            
            "**2. Count and Categorize People:**\n"
            "- Provide an accurate count of people in the scene.\n"
            "- Return a boolean value indicating whether a crowd is present.\n"
            "- Determine the overall level of crowd on a scale from 0 to 10.\n"
            "- Identify demographic patterns if visible (families, tour groups, individuals).\n"
            "- Note accessibility concerns (elderly visitors, children, people with mobility aids).\n\n"
            
            "**3. Identify Safety Concerns:**\n"
            "- Flag potential safety hazards related to overcrowding.\n"
            "- Detect any signs of distress, confusion, or unsafe behavior.\n"
            "- Note objects or environmental factors affecting safety (barriers, narrow passages, weather impacts).\n\n"
            
            "**4. Recognize Peak Congestion Times:**\n"
            "- Identify whether the current time represents peak hour timing (return a boolean value)\n"
            "- Detect the timestamp when crowd density reaches its peak or when significant changes occur.\n\n"
            
            "**5. Assess Need for Intervention**\n"
            "- Determine if police intervention is required (return a boolean value).\n"
            "- If police intervention is required, provide four specific suggestions on how to reduce the crowd (e.g., redirect flows, open additional entry points, station officers at key areas, adjust signal timings).\n"
            
            
            "**7. Follow the Output Format:**\n"
            "Provide your response in the following JSON format:\n"
            """
            {
                "crowd_present": <true/false>,
                "crowd_count": <number>,
                "crowd_level": <number from 0 to 10>,
                "is_peak_hour": <true/false>,
                "police_intervention_required": <true/false>,
                "police_intervention_suggestions": [
                    "<suggestion 1>",
                    "<suggestion 2>",
                    "<suggestion 3>",
                    "<suggestion 4>"
                ]
            }
            """
        )
        
        # Generate analysis
        response = model.generate_content([prompt, image])
        
        return ImageAnalysisResponse(
            status="success",
            analysis={"response": response.text}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@app.get("/", tags=["Info"])
async def root():
    """
    Welcome endpoint with information about the test application
    """
    return {
        "app": "Gemini API Test for Tourist Flow & Safety Monitoring",
        "endpoints": [
            {"path": "/test-gemini/", "method": "GET", "description": "Test Gemini API connection"},
            {"path": "/test-image-analysis/", "method": "POST", "description": "Test image analysis with Gemini"}
        ],
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gemini_test:app", host="0.0.0.0", port=8001, reload=True)
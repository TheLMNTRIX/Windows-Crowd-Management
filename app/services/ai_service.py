import os
import google.generativeai as genai
import json
import concurrent.futures
import asyncio
from PIL import Image, ImageDraw
import numpy as np
from typing import List, Dict, Any
from app.utils.logger import setup_logger
from app.config import settings
from app.services.crowd_counter import CrowdCounter
# from app.services.object_detector import ObjectDetector

logger = setup_logger(__name__)

class AIService:
    def __init__(self):
        # Configure the API
        logger.info("Initializing AI Service with Gemini model")
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        logger.info(f"Using Gemini model: {settings.GEMINI_MODEL}")
        
        # Initialize crowd counter
        self.crowd_counter = CrowdCounter()
        
        # Initialize object detector if needed for additional analysis
        # self.object_detector = ObjectDetector()
    
    def get_crowd_monitoring_prompt(self, location_context: Dict[str, Any] = None, crowd_info: Dict[str, Any] = None) -> str:
        """Return the prompt for crowd monitoring analysis with location context and crowd info"""
        
        # Add location context if available
        location_info = ""
        if location_context:
            location_info = (
                f"Location information: Lat {location_context.get('latitude')}, "
                f"Long {location_context.get('longitude')}, "
                f"Local time {location_context.get('local_time', 'unknown')}\n"
            )
            
        # Add crowd info if available
        crowd_details = ""
        if crowd_info:
            crowd_details = (
                f"Crowd information: Detected {crowd_info.get('crowd_count', 'unknown')} people, "
                f"Crowd level {crowd_info.get('crowd_level', 'unknown')}/10\n"
            )
            
        return (
            "You are an expert crowd analysis system specializing in tourist flow and safety monitoring. "
            "Your task is to provide a structured and detailed analysis of crowd conditions from the video. "
            f"{location_info}"
            f"{crowd_details}"
            "Follow these instructions:\n\n"
            
            "**1. Assess Crowd Density and Flow:**\n"
            "- Identify areas of high, medium, and low crowd density.\n"
            "- Analyze movement patterns, flow directions, and bottlenecks.\n"
            "- Detect unusual congestion points or rapid crowd density changes.\n\n"
            
            "**2. Count and Categorize People:**\n"
            "- I have already counted people for you. The count is provided above.\n"
            "- Return a boolean value indicating whether a crowd is present.\n"
            "- I have already determined the overall crowd level. It's provided above.\n"
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
            "- If police intervention is required, provide four specific suggestions on how to reduce the crowd (e.g., redirect flows, open additional entry points, station officers at key areas, adjust signal timings).\n\n"
            
            "**7. Follow the Output Format:**\n"
            "Provide your response in the following JSON format:\n"
            """
            {
                "crowd_present": <true/false>,
                "crowd_count": "<reuse the count I provided>",
                "crowd_level": "<reuse the crowd level I provided>",
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
        
    def analyze_frames(self, frames: List[Image.Image], timestamps: List[float], location_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze video frames with crowd counter and Gemini AI"""
        logger.info(f"Analyzing {len(frames)} frames with DM-Count model and Gemini AI")
        
        # Step 1: Process frames with crowd counter
        crowd_results = []
        for frame in frames:
            crowd_data = self.crowd_counter.count_crowd(frame)
            crowd_results.append(crowd_data)
        
        # Aggregate crowd counting results
        avg_count = sum(int(result["crowd_count"]) for result in crowd_results) / len(crowd_results)
        max_level = max(int(result["crowd_level"]) for result in crowd_results)
        
        crowd_info = {
            "crowd_count": str(int(round(avg_count))),
            "crowd_level": str(max_level)
        }
        
        # Add timestamp to each frame for context
        annotated_frames = []
        for frame, timestamp in zip(frames, timestamps):
            # Convert numpy array to PIL Image if needed
            if not isinstance(frame, Image.Image):
                pil_frame = Image.fromarray(frame)
            else:
                pil_frame = frame
                
            # Add timestamp
            draw = ImageDraw.Draw(pil_frame)
            time_str = f"Time: {int(timestamp // 60)}m {int(timestamp % 60)}s"
            draw.text((10, 10), time_str, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            
            annotated_frames.append(pil_frame)
        
        # Get prompt with crowd info
        prompt = self.get_crowd_monitoring_prompt(location_context, crowd_info)
        
        logger.info("Sending frames to Gemini for analysis with crowd counting data")
        # Generate content with Gemini
        try:
            response = self.model.generate_content([
                prompt,
                "Please analyze the following frames from a tourist location video for crowd safety and flow:",
                *annotated_frames
            ])
            logger.info("Received response from Gemini")
            
            # Parse the response
            try:
                # Try to extract JSON from response
                result_text = response.text
                logger.debug(f"Raw API response received: {result_text[:100]}...")
                
                # Find JSON in the response (it might be in a code block)
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = result_text[json_start:json_end]
                    result = json.loads(json_str)
                    
                    # Ensure our crowd data is preserved
                    result["crowd_count"] = crowd_info["crowd_count"]
                    result["crowd_level"] = crowd_info["crowd_level"]
                    
                    logger.info(f"Successfully parsed JSON response: crowd_level={result.get('crowd_level', 'unknown')}")
                    return result
                else:
                    # If we can't find JSON, return our crowd data
                    logger.warning("Could not find JSON in the response")
                    return {
                        "crowd_present": True if int(crowd_info["crowd_count"]) > 0 else False,
                        "crowd_count": crowd_info["crowd_count"],
                        "crowd_level": crowd_info["crowd_level"],
                        "is_peak_hour": False,
                        "police_intervention_required": False,
                        "police_intervention_suggestions": [],
                        "raw_response": result_text
                    }
            except Exception as e:
                # Return error with our crowd data
                logger.error(f"Error parsing Gemini response: {str(e)}")
                return {
                    "crowd_present": True if int(crowd_info["crowd_count"]) > 0 else False,
                    "crowd_count": crowd_info["crowd_count"],
                    "crowd_level": crowd_info["crowd_level"],
                    "is_peak_hour": False,
                    "police_intervention_required": False,
                    "police_intervention_suggestions": [],
                    "error": str(e),
                    "raw_response": response.text
                }
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return {
                "crowd_present": True if int(crowd_info["crowd_count"]) > 0 else False,
                "crowd_count": crowd_info["crowd_count"],
                "crowd_level": crowd_info["crowd_level"],
                "is_peak_hour": False,
                "police_intervention_required": False,
                "police_intervention_suggestions": [],
                "error": str(e),
                "raw_response": "API call failed"
            }
    
    async def analyze_frames_parallel(self, frames: List[Image.Image], timestamps: List[float], location_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze video frames in parallel with crowd counter and Gemini AI"""
        logger.info(f"Starting parallel analysis of {len(frames)} frames")
        
        # First run crowd counting on all frames concurrently
        async def count_frame(frame):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: self.crowd_counter.count_crowd(frame)
                )
                return result
        
        # Process frames in parallel for crowd counting
        counting_tasks = [count_frame(frame) for frame in frames]
        crowd_results = await asyncio.gather(*counting_tasks)
        
        # Aggregate crowd results
        avg_count = sum(int(result["crowd_count"]) for result in crowd_results) / len(crowd_results) 
        max_level = max(int(result["crowd_level"]) for result in crowd_results)
        
        crowd_info = {
            "crowd_count": str(int(round(avg_count))),
            "crowd_level": str(max_level)
        }
        
        # Prepare frames with timestamps
        annotated_frames = []
        for frame, timestamp in zip(frames, timestamps):
            if not isinstance(frame, Image.Image):
                pil_frame = Image.fromarray(frame)
            else:
                pil_frame = frame
                
            draw = ImageDraw.Draw(pil_frame)
            time_str = f"Time: {int(timestamp // 60)}m {int(timestamp % 60)}s"
            draw.text((10, 10), time_str, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            annotated_frames.append(pil_frame)
        
        # Update prompt with crowd info
        prompt = self.get_crowd_monitoring_prompt(location_context, crowd_info)
        
        # Process in smaller batches for Gemini
        batch_size = 2
        batches = [annotated_frames[i:i+batch_size] for i in range(0, len(annotated_frames), batch_size)]
        logger.info(f"Created {len(batches)} batches for parallel processing")
        
        # Create processing function for Gemini
        async def process_batch(batch_idx, batch):
            logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda: self.model.generate_content([prompt, "Analyze these frames:", *batch])
                    )
                    logger.info(f"Completed batch {batch_idx+1}/{len(batches)}")
                    return result
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx+1}: {str(e)}")
                raise
        
        # Process batches concurrently with Gemini
        tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
        logger.info("Waiting for all batch processing tasks to complete")
        
        try:
            results = await asyncio.gather(*tasks)
            logger.info("All batches processed successfully")
        except Exception as e:
            logger.error(f"Error during batch processing: {str(e)}")
            # Return with crowd info on error
            return {
                "crowd_present": True if int(crowd_info["crowd_count"]) > 0 else False,
                "crowd_count": crowd_info["crowd_count"],
                "crowd_level": crowd_info["crowd_level"],
                "is_peak_hour": False,
                "police_intervention_required": False,
                "police_intervention_suggestions": [],
                "error": str(e)
            }
        
        # Combine and process results
        logger.info("Combining batch results")
        combined_results = self._combine_batch_results(results, crowd_info)
        return combined_results
    
    async def analyze_frames_with_crowd_info(self, frames: List[Image.Image], timestamps: List[float], 
                                             location_context: Dict[str, Any] = None, 
                                             crowd_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze video frames with Gemini AI using pre-computed crowd info
        
        This is useful when crowd counting has already been done via stream processing
        """
        logger.info(f"Starting Gemini analysis of {len(frames)} frames with provided crowd info")
        
        # Add timestamp to each frame for context
        annotated_frames = []
        for frame, timestamp in zip(frames, timestamps):
            if not isinstance(frame, Image.Image):
                pil_frame = Image.fromarray(frame)
            else:
                pil_frame = frame
                
            # Add timestamp
            draw = ImageDraw.Draw(pil_frame)
            time_str = f"Time: {int(timestamp // 60)}m {int(timestamp % 60)}s"
            draw.text((10, 10), time_str, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            
            annotated_frames.append(pil_frame)
        
        # Get prompt with crowd info
        prompt = self.get_crowd_monitoring_prompt(location_context, crowd_info)
        
        # Process in batches for Gemini
        batch_size = 2
        batches = [annotated_frames[i:i+batch_size] for i in range(0, len(annotated_frames), batch_size)]
        
        # Create processing function for Gemini
        async def process_batch(batch_idx, batch):
            logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda: self.model.generate_content([prompt, "Analyze these frames:", *batch])
                    )
                    logger.info(f"Completed batch {batch_idx+1}/{len(batches)}")
                    return result
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx+1}: {str(e)}")
                raise
        
        # Process batches concurrently with Gemini
        tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
        
        try:
            results = await asyncio.gather(*tasks)
            logger.info("All batches processed successfully")
        except Exception as e:
            logger.error(f"Error during batch processing: {str(e)}")
            # Return with crowd info on error
            return {
                "crowd_present": True if int(crowd_info["crowd_count"]) > 0 else False,
                "crowd_count": crowd_info["crowd_count"],
                "crowd_level": crowd_info["crowd_level"],
                "is_peak_hour": False,
                "police_intervention_required": False,
                "police_intervention_suggestions": [],
                "error": str(e)
            }
        
        # Combine and process results
        combined_results = self._combine_batch_results(results, crowd_info)
        return combined_results

    def _combine_batch_results(self, batch_results, crowd_info=None):
        """Combine and aggregate results from multiple batches"""
        # Extract valid JSON responses
        processed_results = []
        
        for response in batch_results:
            result_text = response.text
            
            try:
                # Find JSON in response
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = result_text[json_start:json_end]
                    result = json.loads(json_str)
                    processed_results.append(result)
                else:
                    logger.warning("Could not find JSON in batch response")
            except Exception as e:
                logger.error(f"Error parsing batch response: {str(e)}")
                
        # No valid results
        if not processed_results:
            # Use crowd info if available
            if crowd_info:
                return {
                    "crowd_present": True if int(crowd_info["crowd_count"]) > 0 else False,
                    "crowd_count": crowd_info["crowd_count"],
                    "crowd_level": crowd_info["crowd_level"],
                    "is_peak_hour": False,
                    "police_intervention_required": False,
                    "police_intervention_suggestions": []
                }
            else:
                return {"error": "No valid results", "crowd_level": "0", "crowd_count": "0"}
        
        # Aggregate results
        logger.info(f"Aggregating results from {len(processed_results)} valid responses")
        
        # Ensure crowd data from DM-Count is preserved
        result = {
            "crowd_present": any(r.get("crowd_present", False) for r in processed_results),
            "crowd_count": crowd_info["crowd_count"] if crowd_info else "0",
            "crowd_level": crowd_info["crowd_level"] if crowd_info else "0",
            "is_peak_hour": any(r.get("is_peak_hour", False) for r in processed_results),
            "police_intervention_required": any(r.get("police_intervention_required", False) for r in processed_results),
            "police_intervention_suggestions": []
        }
        
        # Collect unique suggestions
        all_suggestions = []
        for r in processed_results:
            all_suggestions.extend(r.get("police_intervention_suggestions", []))
        result["police_intervention_suggestions"] = list(set(all_suggestions))[:4]  # Take up to 4 unique suggestions
        
        logger.info(f"Final aggregated result: crowd_level={result['crowd_level']}, intervention_required={result['police_intervention_required']}")
        return result
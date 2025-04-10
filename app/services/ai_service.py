import json
import google.generativeai as genai
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Tuple
import asyncio
import concurrent.futures
from functools import partial
import asyncio
from app.config import settings
from app.utils.logger import setup_logger

# Set up logger for this module
logger = setup_logger(__name__)

class AIService:
    def __init__(self):
        # Configure the API
        logger.info("Initializing AI Service with Gemini model")
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        logger.info(f"Using Gemini model: {settings.GEMINI_MODEL}")
    
    def get_crowd_monitoring_prompt(self, location_context: Dict[str, Any] = None) -> str:
        """Return the prompt for crowd monitoring analysis with location context"""
        logger.debug("Retrieving crowd monitoring prompt with location context")
        
        # Add location context to the prompt if available
        location_info = ""
        if location_context:
            location_info = (
                f"\nLocation Information:\n"
                f"- Latitude: {location_context.get('latitude')}\n"
                f"- Longitude: {location_context.get('longitude')}\n"
                f"- Local Time: {location_context.get('local_time')} (Kolkata IST)\n"
                f"- Full Timestamp: {location_context.get('timestamp')}\n\n"
                f"Use this location and time information when determining if this is a peak hour "
                f"and when assessing crowd conditions. Consider typical patterns for tourist locations "
                f"at this time of day and local conditions.\n\n"
            )
        
        prompt = (
            "You are an expert crowd analysis system specializing in tourist flow and safety monitoring."
            "Your task is to provide a structured and detailed analysis of crowd conditions from the video. "
            f"{location_info}"
            "Follow these instructions:\n\n"
            
            "**1. Assess Crowd Density and Flow:**\n"
            "- Identify areas of high, medium, and low crowd density.\n"
            "- Analyze movement patterns, flow directions, and bottlenecks.\n"
            "- Detect unusual congestion points or rapid crowd density changes.\n\n"
            
            "**2. Count and Categorize People:**\n"
            "- Provide an accurate count of people in the scene. ENSURE TO PROPERLY COUNT THE NUMBER OF PEOPLE, YOU CAN RETURN AN ESTIMATE BUT IT SHOULD BE REAL\n"
            "- If count is more than 5000, return it as '>500', if it is less than 5000, return it as '<500' if it is less than 500, return the exact number\n"
            "- Return a boolean value indicating whether a crowd is present.\n"
            "- Determine the overall level of crowd on a scale from 0 to 10.\n"
            "- Identify demographic patterns if visible (families, tour groups, individuals).\n"
            "- Note accessibility concerns (elderly visitors, children, people with mobility aids).\n\n"
            
            "**3. Identify Safety Concerns:**\n"
            "- Flag potential safety hazards related to overcrowding.\n"
            "- Detect any signs of distress, confusion, or unsafe behavior.\n"
            "- Note objects or environmental factors affecting safety (barriers, narrow passages, weather impacts).\n\n"
            
            "**4. Recognize Peak Congestion Times:**\n"
            "- Considering the location coordinates and current time provided, determine if this represents peak hour timing (return a boolean value).\n"
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
        return prompt
    
    def analyze_frames(self, frames: List[Image.Image], timestamps: List[float], location_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze video frames with Gemini AI"""
        logger.info(f"Analyzing {len(frames)} frames with Gemini AI")
        # Get the prompt with location context
        prompt = self.get_crowd_monitoring_prompt(location_context)
        
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
            # Add white text with black outline
            draw.text((10, 10), time_str, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            
            annotated_frames.append(pil_frame)
        
        logger.info("Sending frames to Gemini for analysis")
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
                    logger.info(f"Successfully parsed JSON response: crowd_level={result.get('crowd_level', 'unknown')}")
                    return result
                else:
                    # If we can't find JSON, return the raw text
                    logger.warning("Could not find JSON in the response")
                    return {"raw_response": result_text, "alert_level": "0"}
            except Exception as e:
                # Return error
                logger.error(f"Error parsing Gemini response: {str(e)}")
                return {"error": str(e), "raw_response": response.text, "alert_level": "0"}
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return {"error": str(e), "raw_response": "API call failed", "alert_level": "0"}
    
    async def analyze_frames_parallel(self, frames: List[Image.Image], timestamps: List[float], location_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze video frames in parallel with Gemini AI"""
        logger.info(f"Starting parallel analysis of {len(frames)} frames")
        prompt = self.get_crowd_monitoring_prompt(location_context)
        
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
        
        # Process in smaller batches (2-3 frames per batch)
        batch_size = 2
        batches = [annotated_frames[i:i+batch_size] for i in range(0, len(annotated_frames), batch_size)]
        logger.info(f"Created {len(batches)} batches for parallel processing")
        
        # Create processing function
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
        
        # Process batches concurrently
        tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
        logger.info("Waiting for all batch processing tasks to complete")
        try:
            results = await asyncio.gather(*tasks)
            logger.info("All batches processed successfully")
        except Exception as e:
            logger.error(f"Error during batch processing: {str(e)}")
            return {"error": str(e), "alert_level": "0"}
        
        # Combine and process results
        logger.info("Combining batch results")
        combined_results = self._combine_batch_results(results)
        return combined_results

    def _combine_batch_results(self, batch_results):
        """Combine and aggregate results from multiple batches"""
        logger.info("Combining results from multiple batches")
        # Logic to combine results from different batches
        all_data = []
        
        for i, response in enumerate(batch_results):
            try:
                result_text = response.text
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = result_text[json_start:json_end]
                    parsed_data = json.loads(json_str)
                    all_data.append(parsed_data)
                    logger.info(f"Successfully parsed batch {i+1} result")
                else:
                    logger.warning(f"Could not find JSON in batch {i+1} result")
            except Exception as e:
                logger.error(f"Error parsing batch {i+1} result: {str(e)}")
        
        # Aggregate results
        if not all_data:
            logger.warning("No valid results found in any batch")
            return {"error": "No valid results", "alert_level": "0"}
        
        # Simple aggregation example
        logger.info(f"Aggregating results from {len(all_data)} valid responses")
        
        # Handle crowd_count as string instead of trying to convert to int
        crowd_counts = [str(d.get("crowd_count", 0)) for d in all_data]
        # Prioritize ">500" over "<500" over numeric values
        if any(">500" in count for count in crowd_counts):
            final_crowd_count = ">500" 
        elif any("<500" in count for count in crowd_counts):
            final_crowd_count = "<500"
        else:
            # For numeric values, use the maximum
            try:
                numeric_counts = [int(count) for count in crowd_counts if count.isdigit()]
                final_crowd_count = str(max(numeric_counts)) if numeric_counts else "0"
            except:
                final_crowd_count = str(crowd_counts[0]) if crowd_counts else "0"
        
        result = {
            "crowd_present": any(d.get("crowd_present", False) for d in all_data),
            "crowd_count": final_crowd_count,
            "crowd_level": max(int(d.get("crowd_level", 0)) for d in all_data),
            "is_peak_hour": any(d.get("is_peak_hour", False) for d in all_data),
            "police_intervention_required": any(d.get("police_intervention_required", False) for d in all_data),
            "police_intervention_suggestions": []
        }
        
        # Collect unique suggestions
        all_suggestions = []
        for d in all_data:
            all_suggestions.extend(d.get("police_intervention_suggestions", []))
        result["police_intervention_suggestions"] = list(set(all_suggestions))[:4]
        
        logger.info(f"Final aggregated result: crowd_level={result['crowd_level']}, intervention_required={result['police_intervention_required']}")
        return result
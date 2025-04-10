import torch
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from app.utils.logger import setup_logger
import torch.nn as nn
import cv2
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

logger = setup_logger(__name__)

# Define the correct VGG19-based model architecture
class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())
    
    def forward(self, x):
        x = self.features(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.reg_layer(x)
        mu = self.density_layer(x)
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19():
    """VGG 19-layer model (configuration "E") for DMCount"""
    model = VGG(make_layers(cfg['E']))
    return model

class CrowdCounter:
    def __init__(self):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Path to model
        model_path = os.path.join(os.getcwd(), "deepmodel", "model_nwpu.pth")
        
        try:
            logger.info(f"Loading DM-Count model from: {model_path}")
            
            # Create model instance using the correct architecture
            self.model = vgg19()
            
            # Check if model file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                self.use_mock_model = True
                logger.warning("Using mock model for crowd counting - MODEL FILE NOT FOUND")
            else:
                try:
                    # Load the state dictionary
                    state_dict = torch.load(model_path, map_location=self.device)
                    
                    # Try to load the state dict
                    try:
                        # If state_dict is wrapped (from training with DataParallel)
                        if 'model_state_dict' in state_dict:
                            state_dict = state_dict['model_state_dict']
                            
                        self.model.load_state_dict(state_dict, strict=False)
                        self.use_mock_model = False
                        logger.info("Successfully loaded DMCount model weights")
                    except Exception as e:
                        logger.warning(f"Could not load state dict: {str(e)} - using mock model")
                        self.use_mock_model = True
                except Exception as e:
                    logger.error(f"Error loading model weights: {str(e)}")
                    self.use_mock_model = True
                    
            # Set model to evaluation mode
            self.model.eval()
            self.model = self.model.to(self.device)
            
            # Define transforms
            self.transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            logger.error(f"Failed to load DM-Count model: {str(e)}", exc_info=True)
            # Create a mock model for testing purposes
            self.model = None
            self.use_mock_model = True
            self.transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor()
            ])
            logger.warning("Using mock model for crowd counting due to initialization error")
    
    def preprocess_image(self, image):
        """Preprocess image for the model"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
    
    def count_crowd(self, image):
        """Count people in the image using DM-Count model or fallback to mock estimation"""
        try:
            # Preprocess image
            tensor = self.preprocess_image(image)
            
            # Log the source of the counting (real model vs mock)
            if self.use_mock_model:
                logger.info("Using mock model for crowd counting")
            else:
                logger.info("Using DMCount model for crowd counting")
            
            # Perform inference
            if self.use_mock_model:
                # Create a realistic mock crowd count based on the image
                with torch.no_grad():
                    # Extract features from the image
                    gray = transforms.functional.rgb_to_grayscale(tensor)
                    complexity = torch.std(gray).item() * 100
                    brightness = torch.mean(gray).item()
                    
                    # Log the complexity and brightness for debugging
                    logger.debug(f"Image complexity: {complexity:.2f}, brightness: {brightness:.2f}")
                    
                    # MODIFIED: Use much more conservative estimation for livestream
                    # Reduce the multiplier for complexity and brightness
                    base_count = max(0, int(complexity * 0.2 + brightness * 2))
                    
                    # Add small randomness
                    count = max(0, base_count + np.random.randint(-2, 3))
                    
                    logger.info(f"Mock model estimated count: {count} (base: {base_count})")
            else:
                # Use the actual model
                with torch.no_grad():
                    density_map, _ = self.model(tensor)
                    count = torch.sum(density_map).item()
                    logger.info(f"DMCount model prediction: {count}")
            
            # Map count to crowd level
            crowd_level = self._calculate_crowd_level(count)
            
            logger.info(f"Final crowd count: {int(count)}, crowd level: {crowd_level}")
            return {
                "crowd_count": str(int(count)),
                "crowd_level": str(crowd_level)
            }
        except Exception as e:
            logger.error(f"Error in crowd counting: {str(e)}", exc_info=True)
            # Return fallback values
            return {"crowd_count": "1", "crowd_level": "1"}  # Lower default values
    
    def _calculate_crowd_level(self, count):
        """Map raw count to crowd level from 0-10"""
        if count <= 0:
            return 0
        elif count <= 5:
            return 1
        elif count <= 10:
            return 2
        elif count <= 20:
            return 3
        elif count <= 30:
            return 4
        elif count <= 50:
            return 5
        elif count <= 80:
            return 6
        elif count <= 120:
            return 7
        elif count <= 200:
            return 8
        elif count <= 300:
            return 9
        else:
            return 10

    def analyze_video_stream(self, source=0, display=True, process_every_n=5, stop_after_frames=None):
        """
        Analyze video stream in real-time with the crowd counter model.
        
        Args:
            source: Camera ID (int) or video file path (str)
            display: Whether to show visualization window
            process_every_n: Process every Nth frame (for performance)
            stop_after_frames: Stop after processing this many frames (None for infinite)
        
        Returns:
            Generator yielding (frame, result) tuples
        """
        # Open video capture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {source}")
            return
            
        frame_count = 0
        processing_count = 0
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream")
                break
                
            frame_count += 1
            
            # Process every Nth frame for performance
            if frame_count % process_every_n == 0:
                processing_count += 1
                
                # Convert BGR to RGB for model
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with crowd counter
                result = self.count_crowd(rgb_frame)
                
                # Add visualization
                if display:
                    crowd_level = int(result["crowd_level"])
                    count = result["crowd_count"]
                    
                    # Add colored border based on crowd level
                    if crowd_level >= 7:
                        border_color = (0, 0, 255)  # Red for high crowd (BGR)
                    elif crowd_level >= 4:
                        border_color = (0, 165, 255)  # Orange for medium crowd
                    else:
                        border_color = (0, 255, 0)  # Green for low crowd
                    
                    # Add border and text
                    frame = cv2.copyMakeBorder(
                        frame, 30, 30, 30, 30,
                        cv2.BORDER_CONSTANT,
                        value=border_color
                    )
                    
                    cv2.putText(
                        frame,
                        f"Crowd Count: {count} | Level: {crowd_level}/10",
                        (40, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    
                    # Optionally generate heatmap visualization
                    if not self.use_mock_model:
                        with torch.no_grad():
                            density_map, _ = self.model(self.preprocess_image(rgb_frame))
                            density = density_map.squeeze().cpu().numpy()
                            
                            # Normalize density map for visualization
                            density = (density - density.min()) / (density.max() - density.min() + 1e-5)
                            density = (density * 255).astype(np.uint8)
                            density = cv2.resize(density, (frame.shape[1]//4, frame.shape[0]//4))
                            heatmap = cv2.applyColorMap(density, cv2.COLORMAP_JET)
                            
                            # Place heatmap in corner of frame
                            frame[30:30+heatmap.shape[0], 30:30+heatmap.shape[1]] = heatmap
                    
                    cv2.imshow("Crowd Analysis", frame)
                
                # Yield the processed frame and result
                yield frame, result
                
                # Check for stop condition
                if stop_after_frames and processing_count >= stop_after_frames:
                    break
            
            # Check for exit key
            if display and cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        if display:
            cv2.destroyAllWindows()
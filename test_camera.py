import cv2
import time

def test_camera(camera_id=0, display_time=10):
    """
    Test a camera by displaying its feed for a few seconds
    
    Args:
        camera_id: ID of the camera to test
        display_time: How long to display the feed in seconds
    """
    print(f"Testing camera with ID {camera_id}...")
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Failed to open camera with ID {camera_id}")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera {camera_id} properties:")
    print(f"- Resolution: {width}x{height}")
    print(f"- FPS: {fps}")
    
    # Display camera feed
    print(f"Displaying camera feed for {display_time} seconds...")
    start_time = time.time()
    
    while time.time() - start_time < display_time:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to capture frame")
            break
        
        # Display frame
        cv2.imshow(f"Camera {camera_id} Test", frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test complete")

if __name__ == "__main__":
    # Test default camera (ID 0)
    test_camera(0)
    
    # Uncomment to test other cameras
    # test_camera(1)
    # test_camera(2)
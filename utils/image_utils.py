import cv2
import os
from datetime import datetime
import numpy as np

def save_screenshot(frame, output_dir="screenshots", event_type="Detection"):
    """Save a screenshot to the specified directory with timestamp and event type"""
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a filename with timestamp and event type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{event_type}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # Save the image
    cv2.imwrite(filepath, frame)
    return filepath

def add_timestamp(frame, text=None):
    """Add timestamp and optional text to the frame"""
    # Make a copy of the frame
    timestamped_frame = frame.copy()
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add timestamp to the bottom left
    cv2.putText(timestamped_frame, timestamp, (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add optional text
    if text:
        cv2.putText(timestamped_frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return timestamped_frame

def resize_image(image, max_width=800):
    """Resize an image while maintaining aspect ratio"""
    height, width = image.shape[:2]
    
    # Check if resizing is needed
    if width > max_width:
        # Calculate new height to maintain aspect ratio
        aspect_ratio = height / width
        new_width = max_width
        new_height = int(new_width * aspect_ratio)
        
        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image
    
    return image
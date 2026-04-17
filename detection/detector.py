import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """Initialize the object detector with YOLOv8 model"""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
    
    def detect(self, frame):
        """Detect objects in the frame"""
        # Run YOLOv8 inference on the frame
        results = self.model(frame, verbose=False)
        
        # Process the results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                # Only include person detections (class 0 in COCO dataset) with sufficient confidence
                if cls == 0 and conf >= self.confidence_threshold:
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf,
                        'class': cls
                    })
        
        return detections
    
    def draw_detections(self, frame, detections, zone_detector=None, bbox_in_zone_fn=None):
        """Draw detection boxes on the frame"""
        for detection in detections:
            bbox = detection['bbox']
            conf = detection['confidence']
            x1, y1, x2, y2 = bbox
            in_zone = False
            if zone_detector:
                if bbox_in_zone_fn:
                    in_zone = bbox_in_zone_fn(bbox, zone_detector)
                else:
                    in_zone = zone_detector.is_in_zone(bbox)
            color = (0, 0, 255) if in_zone else (255, 0, 0)  # Red if in zone, blue otherwise
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{conf:.2f}"
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if in_zone:
                cv2.putText(frame, "Warning: Person Detected in Restricted Area!", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame
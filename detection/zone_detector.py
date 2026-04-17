import cv2
import numpy as np

class ZoneDetector:
    def __init__(self, initial_points=None, frame_width=640, frame_height=480):
        """Initialize the zone detector with optional initial points and frame dimensions"""
        # Store frame dimensions for accurate scaling
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Default zone is a rectangle in the middle of the frame (proportional to frame size)
        default_points = [
            (int(frame_width * 0.2), int(frame_height * 0.2)), 
            (int(frame_width * 0.8), int(frame_height * 0.2)), 
            (int(frame_width * 0.8), int(frame_height * 0.8)), 
            (int(frame_width * 0.2), int(frame_height * 0.8))
        ]
        
        self.zone_points = initial_points if initial_points else default_points
        self.dragging_point = None
        self.point_radius = max(5, min(frame_width, frame_height) // 100)  # Adaptive point radius
    
    def update_frame_dimensions(self, width, height):
        """Update frame dimensions when camera resolution changes"""
        self.frame_width = width
        self.frame_height = height
        self.point_radius = max(5, min(width, height) // 100)
    
    def is_in_zone(self, bbox):
        """Check if any part of the bounding box overlaps with the zone polygon"""
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, self.frame_width))
        y1 = max(0, min(y1, self.frame_height))
        x2 = max(0, min(x2, self.frame_width))
        y2 = max(0, min(y2, self.frame_height))
        
        # Create a polygon from the zone points
        zone_polygon = np.array(self.zone_points, np.int32)
        
        # Create a mask for the zone
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        cv2.fillPoly(mask, [zone_polygon], 255)
        
        # Create a mask for the bounding box
        bbox_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)
        
        # Check for intersection using bitwise AND
        intersection = cv2.bitwise_and(mask, bbox_mask)
        
        # Calculate intersection area
        intersection_area = np.sum(intersection > 0)
        
        # Calculate bounding box area
        bbox_area = (x2 - x1) * (y2 - y1)
        
        # If there's any intersection and it's significant (more than 10% of bbox area)
        return intersection_area > 0 and (intersection_area / bbox_area) > 0.1
    
    def draw_zone(self, frame):
        """Draw the zone polygon and control points on the frame"""
        # Draw the polygon with better visibility
        points = np.array(self.zone_points, np.int32)
        points = points.reshape((-1, 1, 2))
        
        # Draw filled polygon with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
        
        # Draw polygon outline
        cv2.polylines(frame, [points], True, (0, 255, 0), 3)
        
        # Draw the corner points with adaptive size
        for i, point in enumerate(self.zone_points):
            # Draw outer circle
            cv2.circle(frame, point, self.point_radius + 2, (255, 255, 255), -1)
            # Draw inner circle
            cv2.circle(frame, point, self.point_radius, (0, 255, 0), -1)
            # Add point number label with better visibility
            label_size = max(0.4, min(self.frame_width, self.frame_height) / 1000)
            cv2.putText(frame, str(i+1), (point[0]-8, point[1]-15), 
                      cv2.FONT_HERSHEY_SIMPLEX, label_size, (255, 255, 255), 2)
            cv2.putText(frame, str(i+1), (point[0]-8, point[1]-15), 
                      cv2.FONT_HERSHEY_SIMPLEX, label_size, (0, 255, 0), 1)
        
        return frame
    
    def set_zone_points(self, points):
        """Set new zone points"""
        if len(points) == 4:
            self.zone_points = points
    
    def scale_points_from_ui(self, ui_points, ui_width, ui_height):
        """Scale points from UI coordinates to actual frame coordinates"""
        scaled_points = []
        for point in ui_points:
            x_scaled = int(point[0] * self.frame_width / ui_width)
            y_scaled = int(point[1] * self.frame_height / ui_height)
            # Ensure points are within frame bounds
            x_scaled = max(0, min(x_scaled, self.frame_width - 1))
            y_scaled = max(0, min(y_scaled, self.frame_height - 1))
            scaled_points.append((x_scaled, y_scaled))
        return scaled_points
    
    def get_zone_info(self):
        """Get zone information for debugging"""
        return {
            'zone_points': self.zone_points,
            'frame_dimensions': (self.frame_width, self.frame_height),
            'zone_area': self._calculate_zone_area()
        }
    
    def _calculate_zone_area(self):
        """Calculate the area of the zone polygon"""
        if len(self.zone_points) < 3:
            return 0
        
        # Using shoelace formula
        x = [point[0] for point in self.zone_points]
        y = [point[1] for point in self.zone_points]
        
        area = 0
        n = len(x)
        for i in range(n):
            j = (i + 1) % n
            area += x[i] * y[j]
            area -= x[j] * y[i]
        
        return abs(area) / 2
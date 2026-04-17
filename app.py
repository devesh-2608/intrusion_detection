from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import numpy as np
import time
from datetime import datetime, date
import os
import pandas as pd
import threading
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import DateTime, JSON
import json
import io
from pytz import timezone, utc
from queue import Queue

# Import from project modules
from detection.detector import ObjectDetector
from detection.zone_detector import ZoneDetector
from utils.image_utils import save_screenshot, add_timestamp, resize_image
from config import (
    DETECTION_COOLDOWN,
    DEFAULT_ZONE_POINTS,
    CONFIDENCE_THRESHOLD,
    SCREENSHOT_DIR
)

app = Flask(__name__)

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///intrusion.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define database models
class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    camera_index = db.Column(db.Integer, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    zone_points = db.Column(JSON, default=lambda: DEFAULT_ZONE_POINTS)
    created_at = db.Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

class IntrusionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(DateTime, nullable=False, default=datetime.utcnow)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    event_type = db.Column(db.String(50), default='Detection')
    camera = db.relationship('Camera', backref=db.backref('logs', lazy=True))

# Global variables
CAMERAS = {}  # Dictionary to store camera objects
CAMERA_DETECTORS = {}  # Dictionary to store zone detectors for each camera
CAMERA_PREVIOUS_FRAMES = {}  # Dictionary to store previous frames for motion detection
CAMERA_PREVIOUS_PERSON_COUNTS = {}  # Dictionary to store previous person counts
CAMERA_LAST_DETECTION_TIMES = {}  # Dictionary to store last detection times
EVENT_QUEUE = Queue()

# Camera display settings
CAMERAS_PER_PAGE = 4
UI_DISPLAY_WIDTH = 900
UI_DISPLAY_HEIGHT = 675

# Initialize detector
detector = ObjectDetector(confidence_threshold=CONFIDENCE_THRESHOLD)

# Initialize zone points
zone_points = DEFAULT_ZONE_POINTS

# Initialize zone detector
zone_detector = None  # Will be initialized when first frame is captured

CAMERA_CONFIG_FILE = 'cameras.json'

def load_cameras_from_json(json_path):
    """Load camera definitions from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def sync_cameras_with_db(camera_list):
    """Sync the database with the camera list from JSON."""
    existing_cameras = {c.camera_index: c for c in Camera.query.all()}
    for cam in camera_list:
        idx = cam['camera_index']
        # Check if camera exists (by camera_index or RTSP string)
        camera = None
        for c in existing_cameras.values():
            if str(c.camera_index) == str(idx):
                camera = c
                break
        if camera:
            # Update fields
            camera.name = cam['name']
            camera.is_active = cam.get('is_active', True)
            camera.zone_points = cam.get('zone_points', DEFAULT_ZONE_POINTS)
        else:
            # Add new camera
            new_camera = Camera(
                name=cam['name'],
                camera_index=idx,
                is_active=cam.get('is_active', True),
                zone_points=cam.get('zone_points', DEFAULT_ZONE_POINTS)
            )
            db.session.add(new_camera)
    db.session.commit()

def initialize_camera(camera_id, camera_index):
    """Initialize a single camera and its zone detector"""
    try:
        print(f"Initializing camera {camera_id} with index {camera_index}")
        
        # Use different backends for webcam and RTSP
        if isinstance(camera_index, int):
            # For webcam, use DirectShow backend
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            # For RTSP, use default backend
            cap = cv2.VideoCapture(camera_index)
            
        if not cap.isOpened():
            print(f"Warning: Could not open camera {camera_index}")
            return None
        
        # Get actual camera resolution
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera {camera_id} resolution: {frame_width}x{frame_height}")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Only set MJPG codec for webcams
        if isinstance(camera_index, int):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        
        # Test if we can read a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Warning: Could not read frame from camera {camera_index}")
            cap.release()
            return None
            
        # Verify frame is not empty or corrupted
        if frame.size == 0:
            print(f"Warning: Empty frame received from camera {camera_index}")
            cap.release()
            return None
        
        # Initialize zone detector
        camera = db.session.get(Camera, camera_id)
        zone_detector = ZoneDetector(
            initial_points=camera.zone_points if camera else None,
            frame_width=frame_width,
            frame_height=frame_height
        )
        
        # Store camera objects
        CAMERAS[camera_id] = cap
        CAMERA_DETECTORS[camera_id] = zone_detector
        CAMERA_PREVIOUS_FRAMES[camera_id] = None
        CAMERA_PREVIOUS_PERSON_COUNTS[camera_id] = 0
        CAMERA_LAST_DETECTION_TIMES[camera_id] = 0
        
        print(f"Successfully initialized camera {camera_id} (index: {camera_index})")
        return cap
        
    except Exception as e:
        print(f"Error initializing camera {camera_id}: {e}")
        return None

def check_and_update_camera_status():
    """Check all cameras and update their status based on availability."""
    with app.app_context():
        cameras = Camera.query.all()
        print(f"Found {len(cameras)} cameras in the database.")
        for camera in cameras:
            if camera.is_active:
                print(f"Checking camera: {camera.name} (index: {camera.camera_index})")
                is_available = check_camera_availability(camera.camera_index)
                if camera.is_active != is_available:
                    camera.is_active = is_available
                    print(f"Camera {camera.name} status updated to: {'Active' if is_available else 'Inactive'}")
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Error updating camera status: {e}")

def initialize_all_cameras():
    """Initialize all active cameras"""
    with app.app_context():
        active_cameras = Camera.query.filter_by(is_active=True).all()
        print(f"Initializing {len(active_cameras)} active cameras.")
        for camera in active_cameras:
            initialize_camera(camera.id, camera.camera_index)

def detect_motion(frame, camera_id):
    """Detect motion for a specific camera"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if CAMERA_PREVIOUS_FRAMES[camera_id] is None:
        CAMERA_PREVIOUS_FRAMES[camera_id] = gray
        return False, frame
    
    frame_delta = cv2.absdiff(CAMERA_PREVIOUS_FRAMES[camera_id], gray)
    thresh = cv2.threshold(frame_delta, 35, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False
    motion_frame = frame.copy()
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        if (area < 3000 or area > 100000 or w < 50 or h < 50):
            continue
        
        # Check if motion is in zone using bounding box
        motion_bbox = (x, y, x + w, y + h)
        if CAMERA_DETECTORS[camera_id].is_in_zone(motion_bbox):
            motion_detected = True
            cv2.rectangle(motion_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(motion_frame, "Motion", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    CAMERA_PREVIOUS_FRAMES[camera_id] = gray
    return motion_detected, motion_frame

def process_frame(frame, camera_id):
    """Process a single frame for a specific camera"""
    processed_frame = frame.copy()
    detections = detector.detect(processed_frame)
    
    people_in_zone = 0
    for detection in detections:
        if CAMERA_DETECTORS[camera_id].is_in_zone(detection['bbox']):
            people_in_zone += 1
    
    # Draw detections
    processed_frame = detector.draw_detections(processed_frame, detections, CAMERA_DETECTORS[camera_id])
    
    # Detect motion
    motion_detected, motion_frame = detect_motion(frame, camera_id)
    
    # Log events
    current_time = time.time()
    person_count_changed = people_in_zone != CAMERA_PREVIOUS_PERSON_COUNTS[camera_id]
    
    if (person_count_changed or motion_detected) and current_time - CAMERA_LAST_DETECTION_TIMES[camera_id] > DETECTION_COOLDOWN:
        if people_in_zone > CAMERA_PREVIOUS_PERSON_COUNTS[camera_id]:
            event_type = "Entry"
            message = f"Person entered the zone in camera {camera_id}"
        elif people_in_zone < CAMERA_PREVIOUS_PERSON_COUNTS[camera_id]:
            event_type = "Exit"
            message = f"Person left the zone in camera {camera_id}"
        elif motion_detected:
            event_type = "Significant Motion"
            message = f"Significant motion detected in zone for camera {camera_id}"
        else:
            event_type = "Change"
            message = f"Zone activity detected in camera {camera_id}"
        
        # Save screenshot and log
        timestamped_frame = add_timestamp(processed_frame)
        image_path = save_screenshot(timestamped_frame, SCREENSHOT_DIR, event_type)
        
        log = IntrusionLog(
            camera_id=camera_id,
            image_path=image_path,
            event_type=event_type
        )
        with app.app_context():
            db.session.add(log)
            db.session.commit()
        
        # Add event to queue for SSE
        EVENT_QUEUE.put({
            'type': event_type,
            'message': message,
            'camera_id': camera_id,
            'screenshot': True
        })
        
        CAMERA_LAST_DETECTION_TIMES[camera_id] = current_time
        print(f"Event logged: {event_type} for camera {camera_id} at {datetime.now()}")
    
    CAMERA_PREVIOUS_PERSON_COUNTS[camera_id] = people_in_zone
    return processed_frame, people_in_zone > 0 or motion_detected

def generate_frames(camera_id):
    """Generate frames for a specific camera"""
    cap = CAMERAS.get(camera_id)
    if not cap:
        print(f"Camera {camera_id} not initialized")
        return
    
    while True:
        try:
            success, frame = cap.read()
            if not success or frame is None:
                print(f"Failed to read frame from camera {camera_id}")
                # Try to reinitialize the camera
                camera = db.session.get(Camera, camera_id)
                if camera and camera.is_active:
                    cap = cv2.VideoCapture(camera.camera_index, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        print(f"Failed to reinitialize camera {camera_id}")
                        break
                    CAMERAS[camera_id] = cap
                    continue
                else:
                    break
            
            # Verify frame is valid
            if frame.size == 0:
                print(f"Empty frame received from camera {camera_id}")
                continue

            # Debug: Save the first raw frame
            if not hasattr(generate_frames, "debug_saved"):
                os.makedirs("test_output", exist_ok=True)
                cv2.imwrite("test_output/flask_first_frame.jpg", frame)
                generate_frames.debug_saved = True
            
            # Process the frame
            processed_frame, _ = process_frame(frame, camera_id)

            # Debug: Save the first processed frame
            if not hasattr(generate_frames, "debug_saved_processed"):
                cv2.imwrite("test_output/flask_first_processed_frame.jpg", processed_frame)
                generate_frames.debug_saved_processed = True
            
            # Resize frame for display
            display_frame = cv2.resize(processed_frame, (UI_DISPLAY_WIDTH, UI_DISPLAY_HEIGHT))
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            print(f"Error processing frame for camera {camera_id}: {e}")
            continue

def check_camera_availability(camera_index):
    """Check if a camera is available and can be opened"""
    try:
        if isinstance(camera_index, str) and camera_index.startswith('rtsp://'):
            # For RTSP cameras, try to open the stream
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return False
            ret, _ = cap.read()
            cap.release()
            return ret
        else:
            # For local cameras, try to open the device
            cap = cv2.VideoCapture(int(camera_index))
            if not cap.isOpened():
                return False
            ret, _ = cap.read()
            cap.release()
            return ret
    except Exception as e:
        print(f"Error checking camera {camera_index}: {e}")
        return False

@app.route('/')
def index():
    """Show grid view of cameras"""
    page = request.args.get('page', 1, type=int)
    
    # Get all cameras from database (active and inactive)
    all_cameras = Camera.query.order_by(Camera.id).all()
    
    # Calculate pagination
    total_cameras = len(all_cameras)
    total_pages = (total_cameras + CAMERAS_PER_PAGE - 1) // CAMERAS_PER_PAGE
    
    # Get cameras for current page
    start_idx = (page - 1) * CAMERAS_PER_PAGE
    end_idx = start_idx + CAMERAS_PER_PAGE
    current_page_cameras = all_cameras[start_idx:end_idx]
    
    # Create pagination object
    class Pagination:
        def __init__(self, page, total_pages):
            self.page = page
            self.pages = total_pages
            self.has_prev = page > 1
            self.has_next = page < total_pages
            self.prev_num = page - 1
            self.next_num = page + 1
            
        def iter_pages(self):
            for p in range(1, self.pages + 1):
                yield p
    
    pagination = Pagination(page, total_pages)
    
    return render_template('index.html', 
                         cameras=current_page_cameras,
                         pagination=pagination)

@app.route('/camera/<int:camera_id>')
def camera_view(camera_id):
    """Show single camera view with zone configuration"""
    camera = Camera.query.get_or_404(camera_id)
    return render_template('camera.html', camera=camera)

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    """Video feed for a specific camera"""
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_zone/<int:camera_id>', methods=['POST'])
def update_zone(camera_id):
    """Update zone points for a specific camera"""
    try:
        data = request.get_json()
        points = data.get('points')
        
        if not points or len(points) != 4:
            return jsonify({'status': 'error', 'message': 'Invalid zone points'}), 400
        
        if not all(isinstance(x, list) and len(x) == 2 and 
                  all(isinstance(i, (int, float)) for i in x) for x in points):
            return jsonify({'status': 'error', 'message': 'Invalid point format'}), 400
        
        camera = Camera.query.get_or_404(camera_id)
        zone_detector = CAMERA_DETECTORS.get(camera_id)
        
        if not zone_detector:
            return jsonify({'status': 'error', 'message': 'Zone detector not initialized'}), 400
        
        # Scale points from UI coordinates to actual camera coordinates
        scaled_points = zone_detector.scale_points_from_ui(
            points, UI_DISPLAY_WIDTH, UI_DISPLAY_HEIGHT
        )
        
        # Update camera zone points in database
        camera.zone_points = scaled_points
        db.session.commit()
        
        # Update zone detector
        zone_detector.set_zone_points(scaled_points)
        
        print(f"Zone updated for camera {camera_id} with points: {scaled_points}")
        return jsonify({
            'status': 'success',
            'scaled_points': scaled_points,
            'zone_info': zone_detector.get_zone_info()
        })
        
    except Exception as e:
        print(f"Error updating zone for camera {camera_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_zone_info/<int:camera_id>')
def get_zone_info(camera_id):
    """Get current zone information for a specific camera"""
    zone_detector = CAMERA_DETECTORS.get(camera_id)
    if not zone_detector:
        return jsonify({'error': 'Zone detector not initialized'})
    
    return jsonify(zone_detector.get_zone_info())

@app.template_filter('localtime')
def localtime_filter(value):
    if value is None:
        return ''
    local_tz = timezone('Asia/Kolkata')
    if value.tzinfo is None:
        value = utc.localize(value)
    return value.astimezone(local_tz).strftime('%Y-%m-%d %H:%M:%S')

@app.route('/logs')
def view_logs():
    category = request.args.get('category', 'all')
    date_filter = request.args.get('date')
    query = IntrusionLog.query
    
    if category != 'all':
        query = query.filter_by(event_type=category)
    if date_filter:
        query = query.filter(db.func.date(IntrusionLog.timestamp) == date_filter)
    
    logs = query.order_by(IntrusionLog.timestamp.desc()).all()
    
    for log in logs:
        log.image_path = os.path.basename(log.image_path)
    
    return render_template('logs.html', logs=logs)

@app.route('/screenshots/<path:filename>')
def serve_screenshot(filename):
    return send_file(os.path.join(SCREENSHOT_DIR, filename))

@app.route('/export_logs')
def export_logs():
    category = request.args.get('category', 'all')
    date_filter = request.args.get('date')
    query = IntrusionLog.query
    
    if category != 'all':
        query = query.filter_by(event_type=category)
    if date_filter:
        query = query.filter(db.func.date(IntrusionLog.timestamp) == date_filter)
    
    logs = query.order_by(IntrusionLog.timestamp.desc()).all()
    data = []
    
    for log in logs:
        local_tz = timezone('Asia/Kolkata')
        ts = log.timestamp
        if ts.tzinfo is None:
            ts = utc.localize(ts)
        local_ts = ts.astimezone(local_tz).strftime('%Y-%m-%d %H:%M:%S')
        data.append({
            'ID': log.id,
            'Timestamp': local_ts,
            'Camera ID': log.camera_id,
            'Event Type': log.event_type,
            'Image Path': os.path.basename(log.image_path)
        })
    
    df = pd.DataFrame(data)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return Response(
        output,
        mimetype='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename=intrusion_logs.csv'
        }
    )

@app.template_filter('basename')
def basename_filter(path):
    return os.path.basename(path)

@app.route('/events')
def events():
    def generate():
        while True:
            if not EVENT_QUEUE.empty():
                event = EVENT_QUEUE.get()
                yield f"data: {json.dumps(event)}\n\n"
            time.sleep(0.1)
    
    return Response(generate(), mimetype='text/event-stream')

# Initialize database and cameras on startup
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        camera_list = load_cameras_from_json(CAMERA_CONFIG_FILE)
        sync_cameras_with_db(camera_list)
        check_and_update_camera_status()
        initialize_all_cameras()
    app.run(debug=False, host='0.0.0.0', port=4000, use_reloader=False)
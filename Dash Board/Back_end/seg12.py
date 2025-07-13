import os
import cv2
import time
import csv
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from torchvision import transforms, models
from ultralytics import YOLO
from torch import nn
from gpio_controller import turn_on_device, turn_off_device, cleanup_gpio
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging
from flask import Flask, jsonify, Response, render_template, request, session, send_from_directory
from flask_cors import CORS
import threading
import json
import socket

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Serve frontend HTML files
@app.route('/')
def serve_index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/device-control.html')
def serve_device_control():
    return send_from_directory('../frontend', 'device-control.html')

# Serve static assets (JS, CSS)
@app.route('/<path:filename>')
def serve_static_file(filename):
    return send_from_directory('../frontend', filename)

# Global variables for system state
current_stats = {
    'people_count': 0,
    'density': 0.0,
    'device_status': 'off',
    'crowd_level': 'unknown',
    'crowd_confidence': 0.0,
    'tracked_ids': [],
    'tracking_info': [],
    'processing_time': 0,
    'fps': 0,
    'frame_count': 0
}

detection_running = False
frame_buffer = None
cap = None

# In-memory storage for devices (you can replace with database)
devices = {
    'fan': {'name': 'Ceiling Fan', 'room': 'Living Room', 'speed': 0, 'state': False},
    'light': {'name': 'Smart Light', 'room': 'Living Room', 'brightness': 0, 'state': False},
    'tv': {'name': 'Smart TV', 'room': 'Living Room', 'volume': 0, 'state': False}
}

# Device control history
device_history = []

# Configuration constants
CAMERA_ID = 0
CAMERA_FOV_M2 = 100  # Camera field of view in square meters
DENSITY_THRESHOLD = 0.05
CSV_FILE = "density_log.csv"
DATASET_DIR = "dataset"
OUTPUT_DIR = "output_results"
YOLO_MODEL_PATH = "yolov8n.pt"
RESNET_MODEL_PATH = "model/resnet50_density_model.pth"
MIN_CONFIDENCE = 0.3
FRAME_SKIP = 2
FORCE_GPU = True
BATCH_SIZE = 4
USE_RESNET = True

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('model', exist_ok=True)

# Initialize CSV file
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Camera ID", "People Count", "Density", "Device Status", 
                         "Crowd Density", "Tracked IDs", "Processing Time (ms)"])

def check_mps_availability():
    """Check if Apple Silicon GPU is available"""
    if torch.backends.mps.is_available():
        logger.info("MPS is available. Using Apple Silicon GPU!")
        return torch.device("mps")
    else:
        logger.info("MPS not available. Falling back to CPU.")
        return torch.device("cpu")

# Setup torch device
torch_device = check_mps_availability()
logger.info(f"Using device: {torch_device}")

# Load YOLO model
yolo_model = YOLO(YOLO_MODEL_PATH)
if torch_device.type == 'mps':
    yolo_model.to(torch_device)
    logger.info("YOLO model moved to GPU")

class DensityResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet50(weights='IMAGENET1K_V2')
        self.base_model.fc = nn.Identity()
        self.base_model.avgpool = nn.Identity()
        self.density_head = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.density_head(x)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x

class CrowdDensityClassifier(nn.Module):
    def __init__(self, density_model):
        super().__init__()
        self.density_model = density_model
        for param in self.density_model.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        density_map = self.density_model(x)
        crowd_level = self.classifier(density_map)
        return density_map, crowd_level

def load_trained_model(model_path, device):
    """Load trained ResNet model"""
    try:
        model = DensityResNet()
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            logger.info("Loaded checkpoint format")
        else:
            state_dict = checkpoint
            logger.info("Loaded direct state dict format")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info("DensityResNet model loaded successfully")
        classifier_model = CrowdDensityClassifier(model)
        classifier_model.to(device)
        classifier_model.eval()
        return model, classifier_model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

# Load ResNet model
resnet_model = None
crowd_classifier = None

if USE_RESNET:
    try:
        logger.info("Loading trained DensityResNet model...")
        resnet_model, crowd_classifier = load_trained_model(RESNET_MODEL_PATH, torch_device)
        if resnet_model is None:
            logger.warning("Could not load ResNet model, continuing without it")
            USE_RESNET = False
        else:
            logger.info("ResNet density model loaded successfully")
    except FileNotFoundError:
        logger.warning("ResNet model not found, continuing without it")
        USE_RESNET = False
    except Exception as e:
        logger.warning(f"Error loading ResNet model: {e}, continuing without it")
        USE_RESNET = False

# Setup transforms
resnet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize DeepSORT tracker
tracker = DeepSort(
    max_age=50,
    n_init=3,
    max_cosine_distance=0.4,
    nn_budget=None,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=torch_device.type == 'mps',
    embedder_wts=None,
    polygon=False,
    today=None
)

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter_area / float(box1_area + box2_area - inter_area)

def analyze_crowd_density(frame, model, classifier, transform, device):
    """Analyze crowd density using ResNet"""
    if model is None or classifier is None:
        return 0.0, "Unknown", 0.0
    
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(frame_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            density_map, crowd_level = classifier(input_tensor)
            
            total_count = torch.sum(density_map).item()
            
            crowd_probs = crowd_level[0]
            crowd_class = torch.argmax(crowd_probs).item()
            crowd_confidence = torch.max(crowd_probs).item()
            
            crowd_labels = ["Low", "Medium", "High", "Very High"]
            crowd_label = crowd_labels[crowd_class]
            
            return total_count, crowd_label, crowd_confidence
            
    except Exception as e:
        logger.warning(f"Error in crowd analysis: {e}")
        return 0.0, "Error", 0.0

def process_frame():
    """Main frame processing function"""
    global current_stats, frame_buffer, cap
    
    if not detection_running or cap is None:
        return
    
    start_time = time.time()
    ret, frame = cap.read()
    
    if not ret:
        logger.warning("Failed to read frame from camera")
        return
    
    try:
        # YOLO detection for people
        results = yolo_model(frame, verbose=False, conf=MIN_CONFIDENCE)
        people_boxes = []
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if yolo_model.names[cls] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Validate bounding box
                    if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                        continue
                    
                    people_boxes.append((x1, y1, x2, y2, conf))
        
        # Format detections for DeepSORT
        formatted_detections = []
        for x1, y1, x2, y2, conf in people_boxes:
            formatted_detections.append(([x1, y1, x2, y2], conf, 0))
        
        # Update tracker
        tracks = tracker.update_tracks(formatted_detections, frame=frame)
        
        # Process tracked objects
        tracked_people = []
        active_track_ids = []
        tracking_info = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            active_track_ids.append(track_id)
            
            # Find best matching detection
            best_match = None
            best_iou = 0
            for det in people_boxes:
                det_box = det[:4]
                iou = calculate_iou(ltrb, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = det
            
            if best_match and best_iou > 0.3:
                conf = best_match[4]
            else:
                conf = 0.7
            
            tracked_people.append((x1, y1, x2, y2, track_id, conf))
            
            # Add to tracking info for frontend
            tracking_info.append({
                'track_id': track_id,
                'position': f"({x1},{y1})",
                'confidence': round(conf, 2),
                'status': 'Confirmed'
            })
        
        # Calculate density and device status
        people_count = len(tracked_people)
        density = people_count / CAMERA_FOV_M2
        device_status = "on" if people_count > 0 else "off"

        # Control GPIO devices
        try:
            if device_status == "on":
                turn_on_device()
        # Control all devices based on their type
                for device_id, device in devices.items():
                    device_type = device.get('type', device_id.split('-')[0])  # Get type from config or ID
            
                    if device_type == 'fan' or device_id == 'fan':
                        speed = min(100, max(30, int(density * 200)))
                        devices[device_id]['state'] = True
                        devices[device_id]['speed'] = speed
                    elif device_type == 'light' or device_id == 'light':
                        brightness = min(100, max(40, int(density * 250)))
                        devices[device_id]['state'] = True
                        devices[device_id]['brightness'] = brightness
                    elif device_type == 'tv' or device_id == 'tv':
                        volume = min(100, max(10, int(density * 150)))
                        devices[device_id]['state'] = True
                        devices[device_id]['volume'] = volume
                    elif device_type == 'ac':
                        temperature = min(30, max(18, int(20 + density * 50)))
                        devices[device_id]['state'] = True
                        devices[device_id]['temperature'] = temperature
            else:
                turn_off_device()
                # Turn off all devices
                for device_id, device in devices.items():
                    devices[device_id]['state'] = False
                    if 'speed' in devices[device_id]:
                        devices[device_id]['speed'] = 0
                    if 'brightness' in devices[device_id]:
                        devices[device_id]['brightness'] = 0
                    if 'volume' in devices[device_id]:
                        devices[device_id]['volume'] = 0
                    if 'temperature' in devices[device_id]:
                        devices[device_id]['temperature'] = 20
        except Exception as e:
            logger.warning(f"GPIO control error: {e}")
        
        # Analyze crowd density with ResNet
        crowd_level = "Unknown"
        crowd_confidence = 0.0
        if USE_RESNET and resnet_model is not None:
            _, crowd_level, crowd_confidence = analyze_crowd_density(
                frame, resnet_model, crowd_classifier, resnet_transform, torch_device
            )
        
        # Calculate processing time and FPS
        processing_time = (time.time() - start_time) * 1000
        fps = 1000 / processing_time if processing_time > 0 else 0
        
        # Update global stats
        current_stats.update({
            'people_count': people_count,
            'density': round(density, 4),
            'device_status': device_status,
            'crowd_level': crowd_level,
            'crowd_confidence': round(crowd_confidence, 2),
            'tracked_ids': active_track_ids,
            'tracking_info': tracking_info,
            'processing_time': round(processing_time, 2),
            'fps': round(fps, 1),
            'frame_count': current_stats['frame_count'] + 1
        })
        
        # Draw tracking results on frame
        for x1, y1, x2, y2, track_id, conf in tracked_people:
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw track ID and confidence
            label = f"ID:{track_id} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw system info on frame
        cv2.putText(frame, f"People: {people_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Density: {density:.4f} ppl/m² | Device: {device_status.upper()}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Crowd: {crowd_level} ({crowd_confidence:.2f})", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | Tracks: {len(active_track_ids)}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Update frame buffer
        frame_buffer = frame.copy()
        
        # Log to CSV
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        tracked_ids_str = ",".join(map(str, active_track_ids)) if active_track_ids else "None"
        
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                f"Camera {CAMERA_ID}",
                people_count,
                f"{density:.4f}",
                device_status.upper(),
                f"{crowd_level} ({crowd_confidence:.2f})",
                tracked_ids_str,
                f"{processing_time:.2f}"
            ])
        
        # Send person count to Raspberry Pi
        send_person_count_to_pi(people_count)
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")

def detection_loop():
    """Main detection loop running in separate thread"""
    global detection_running, cap
    
    logger.info("Starting detection loop")
    
    while detection_running:
        try:
            process_frame()
            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            logger.error(f"Error in detection loop: {e}")
            time.sleep(1)
    
    logger.info("Detection loop stopped")

def generate_frames():
    """Generate frames for video streaming"""
    global frame_buffer
    
    while True:
        if frame_buffer is not None and detection_running:
            try:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame_buffer, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.error(f"Error generating frame: {e}")
        else:
            # Send placeholder image when detection is not running
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera Feed Stopped", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            _, buffer = cv2.imencode('.jpg', placeholder)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/api/detection_status')
def detection_status():
    """Get current detection status and stats with device integration"""
    # Calculate overall device status
    any_device_on = any(d.get('state') for d in devices.values()) 
    return jsonify({
        'running': detection_running,
        'stats': {
            **current_stats,
            'devices': devices,
            'device_status': 'on' if any_device_on else current_stats.get('device_status', 'off'),
            'total_devices': len(devices),
            'active_devices': sum(1 for d in devices.values() if d.get('state'))
        }
    })


@app.route('/api/start_detection', methods=['POST'])
def start_detection():
    """Start detection process"""
    global detection_running, cap
    
    try:
        if not detection_running:
            # Initialize camera
            cap = cv2.VideoCapture(CAMERA_ID)
            if not cap.isOpened():
                return jsonify({'error': 'Cannot access camera'}), 500
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            detection_running = True
            
            # Start detection thread
            detection_thread = threading.Thread(target=detection_loop, daemon=True)
            detection_thread.start()
            
            logger.info("Detection started successfully")
            return jsonify({'status': 'Detection started'})
        else:
            return jsonify({'status': 'Detection already running'})
            
    except Exception as e:
        logger.error(f"Error starting detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    """Stop detection process"""
    global detection_running, cap, frame_buffer
    
    try:
        if detection_running:
            detection_running = False
            
            # Release camera
            if cap is not None:
                cap.release()
                cap = None
            
            # Clear frame buffer
            frame_buffer = None
            
            # Reset stats
            current_stats.update({
                'people_count': 0,
                'density': 0.0,
                'device_status': 'off',
                'crowd_level': 'unknown',
                'crowd_confidence': 0.0,
                'tracked_ids': [],
                'tracking_info': [],
                'processing_time': 0,
                'fps': 0
            })
            
            # Turn off devices
            try:
                turn_off_device()
            except:
                pass
            
            logger.info("Detection stopped successfully")
            return jsonify({'status': 'Detection stopped'})
        else:
            return jsonify({'status': 'Detection not running'})
            
    except Exception as e:
        logger.error(f"Error stopping detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/system_info')
def system_info():
    """Get system information"""
    return jsonify({
        'device': str(devices),
        'yolo_model': YOLO_MODEL_PATH,
        'resnet_enabled': USE_RESNET,
        'camera_id': CAMERA_ID,
        'camera_fov': CAMERA_FOV_M2,
        'density_threshold': DENSITY_THRESHOLD
    })

# Device Control API Routes (Added from pasted code)
@app.route('/device-control')
def device_control():
    """Render the device control page"""
    return render_template('device-control.html')

@app.route('/api/devices', methods=['GET'])
def get_devices():
    """Get all devices and their current states"""
    try:
        return jsonify({
            'status': 'success',
            'devices': devices,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    

@app.route('/api/device/toggle', methods=['POST'])
def toggle_device():
    """Toggle device on/off state"""
    try:
        data = request.get_json()
        device_id = data.get('device')
        new_state = data.get('state')

        if device_id not in devices:
            return jsonify({
                'status': 'error',
                'message': 'Device not found'
            }), 404
        
        # Update device state
        devices[device_id]['state'] = new_state
        devices[device_id]['last_updated'] = datetime.now().isoformat()

        # If turning off, reset control values
        if not new_state:
            if 'speed' in devices[device_id]:
                devices[device_id]['speed'] = 0
            if 'brightness' in devices[device_id]:
                devices[device_id]['brightness'] = 0
            if 'volume' in devices[device_id]:
                devices[device_id]['volume'] = 0
            if 'temperature' in devices[device_id]:
                devices[device_id]['temperature'] = 0

        # Log the action
        device_history.append({
            'device_id': device_id,
            'action': 'toggle',
            'state': new_state,
            'timestamp': datetime.now().isoformat(),
            'device_name': devices[device_id]['name']
        })
        
        return jsonify({
            'status': 'success',
            'device': device_id,
            'new_state': new_state,
            'message': f"Device {devices[device_id]['name']} turned {'on' if new_state else 'off'}",
            'timestamp': devices[device_id]['last_updated']
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/device/control', methods=['POST'])
def control_device():
    """Update device control settings (brightness, speed, volume, etc.)"""
    try:
        data = request.get_json()
        device_id = data.get('device')
        control_type = data.get('control')
        value = data.get('value')
        
        if device_id not in devices:
            return jsonify({
                'status': 'error',
                'message': 'Device not found'
            }), 404
        
        # Validate control type and update device
        valid_controls = {
            'speed': ['fan'],
            'brightness': ['light'],
            'volume': ['tv'],
            'temperature': ['ac']
        }
        
        # Extract base device type from device_id (handles dynamic IDs like 'fan-123456')
        base_device_type = device_id.split('-')[0]
        valid_controls = {
            'speed': ['fan'],
            'brightness': ['light'],
            'volume': ['tv'],
            'temperature': ['ac']
        }

        if control_type not in valid_controls or base_device_type not in valid_controls[control_type]:
            return jsonify({
                'status': 'error',
                'message': f"Invalid control type '{control_type}' for device type '{base_device_type}'"
            }), 400

        # Validate value
        if not isinstance(value, (int, float)):
            return jsonify({
                'status': 'error',
                'message': f"Invalid value type for control '{control_type}'. Expected int or float."
            }), 400

        # Update the control value
        devices[device_id][control_type] = value
        devices[device_id]['last_updated'] = datetime.now().isoformat()

        # Auto-turn on device if control value > 0
        if value > 0:
            devices[device_id]['state'] = True

        # Log the action
        device_history.append({
            'device_id': device_id,
            'action': 'control_update',
            'control_type': control_type,
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'device_name': devices[device_id]['name']
        })
        
        return jsonify({
            'status': 'success',
            'device': device_id,
            'control': control_type,
            'value': value,
            'message': f"{devices[device_id]['name']} {control_type} set to {value}%",
            'timestamp': devices[device_id]['last_updated']
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/device/add', methods=['POST'])
def add_device():
    """Add a new device to the system"""
    try:
        data = request.get_json()
        device_type = data.get('type')
        device_name = data.get('name')
        device_room = data.get('room')
        
        if not all([device_type, device_name, device_room]):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: type, name, room'
            }), 400
        
        # Generate unique device ID
        import time
        device_id = f"{device_type}-{int(time.time())}"
        
        # Create device configuration based on type
        device_config = {
            'name': device_name,
            'state': False,
            'room': device_room,
            'type': device_type,
            'last_updated': datetime.now().isoformat(),
            'created_at': datetime.now().isoformat()
        }
        
        # Add type-specific controls
        if device_type == 'fan':
            device_config['speed'] = 1
        elif device_type == 'light':
            device_config['brightness'] = 0
        elif device_type == 'tv':
            device_config['volume'] = 0
        elif device_type == 'ac':
            device_config['temperature'] = 20  # Default temperature
        
        # Add device to storage
        devices[device_id] = device_config
        
        # Log the action
        device_history.append({
            'device_id': device_id,
            'action': 'device_added',
            'device_type': device_type,
            'device_name': device_name,
            'room': device_room,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'status': 'success',
            'device_id': device_id,
            'device': device_config,
            'message': f"Device '{device_name}' added successfully"
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/device/remove/<device_id>', methods=['DELETE'])
def remove_device(device_id):
    """Remove a device from the system"""
    try:
        if device_id not in devices:
            return jsonify({
                'status': 'error',
                'message': 'Device not found'
            }), 404

        device_name = devices[device_id]['name']
        del devices[device_id]

        # Log the action
        device_history.append({
            'device_id': device_id,
            'action': 'device_removed',
            'device_name': device_name,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'status': 'success',
            'message': f"Device '{device_name}' removed successfully"
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/device/history', methods=['GET'])
def get_device_history():
    """Get device control history"""
    try:
        # Get last N entries (default 50)
        limit = request.args.get('limit', 50, type=int)
        
        return jsonify({
            'status': 'success',
            'history': device_history[-limit:],
            'total_entries': len(device_history)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/device/stats', methods=['GET'])
def get_device_stats():
    """Get device statistics"""
    try:
        stats = {
            'total_devices': len(devices),
            'active_devices': sum(1 for d in devices.values() if d.get('state')),
            'inactive_devices': sum(1 for d in devices.values() if not d.get('state')),
            'devices_by_room': {},
            'devices_by_type': {}
        }
        
        # Count devices by room and type
        for d in devices.values():
            room = d.get('room', 'Unknown')
            device_type = d.get('type', 'Unknown')

            stats['devices_by_room'][room] = stats['devices_by_room'].get(room, 0) + 1
            stats['devices_by_type'][device_type] = stats['devices_by_type'].get(device_type, 0) + 1
        
        return jsonify({
            'status': 'success',
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/density')
def get_density():
    """Get current density from live detection stats"""
    return jsonify({
        "density": current_stats.get('density', 0),
        "people_count": current_stats.get('people_count', 0),
        "device_status": current_stats.get('device_status', 'off'),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/device/status/<device_id>', methods=['GET'])
def get_device_status(device_id):
    """Get specific device status"""
    try:
        if device_id not in devices:
            return jsonify({
                'status': 'error',
                'message': 'Device not found'
            }), 404
        
        return jsonify({
            'status': 'success',
            'device_id': device_id,
            'device': devices[device_id]
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

def send_person_count_to_pi(count):
    HOST = '192.168.137.143'  # Replace with your Pi's IP address
    PORT = 5001
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(str(count).encode(), (HOST, PORT))

if __name__ == '__main__':
    try:
        logger.info("Starting Crowd Management System Backend")
        logger.info(f"Using devices: {devices}")
        logger.info(f"ResNet enabled: {USE_RESNET}")
        logger.info("Starting Flask server on http://0.0.0.0:5000")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Cleanup
        detection_running = False
        if cap is not None:
            cap.release()
        cleanup_gpio()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")

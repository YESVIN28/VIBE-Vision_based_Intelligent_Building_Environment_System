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


CAMERA_ID = 0
CAMERA_FOV_M2 = 10 * 10
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('model', exist_ok=True)

if os.path.dirname(RESNET_MODEL_PATH):
    os.makedirs(os.path.dirname(RESNET_MODEL_PATH), exist_ok=True)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Camera ID", "People Count", "Density", "Device Status", 
                         "Crowd Density", "Tracked IDs", "Processing Time (ms)"])

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter_area / float(box1_area + box2_area - inter_area)

def check_mps_availability():
    if torch.backends.mps.is_available():
        print("MPS is available. Using Apple Silicon GPU!")
        return torch.device("mps")
    else:
        print("MPS not available. Falling back to CPU.")
        return torch.device("cpu")

device = check_mps_availability()

gpu_available = check_mps_availability()
device = torch.device('mps' if gpu_available and FORCE_GPU else 'cpu')
logger.info(f"Using device: {device}")

yolo_model = YOLO(YOLO_MODEL_PATH)
if device.type == 'mps':
    yolo_model.to(device)
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
        logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
        
        classifier_model = CrowdDensityClassifier(model)
        classifier_model.to(device)
        classifier_model.eval()
        
        return model, classifier_model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

USE_RESNET = True

def create_and_train_resnet_model():
    logger.warning("This function is not needed - your model is already trained!")
    logger.info("Your trained model is at: " + RESNET_MODEL_PATH)
    logger.info("It was trained using custom_dataset.py with ShanghaiTech dataset")
    return None

resnet_model = None
crowd_classifier = None

if USE_RESNET:
    try:
        logger.info("Loading trained DensityResNet model...")
        
        resnet_model, crowd_classifier = load_trained_model(RESNET_MODEL_PATH, device)
        
        if resnet_model is None:
            raise Exception("Could not load the trained model")
            
        logger.info("ResNet density model loaded successfully")
        if device.type == 'mps':
            logger.info("ResNet model moved to GPU")
            
    except FileNotFoundError:
        logger.error("ResNet model not found at: " + RESNET_MODEL_PATH)
        logger.info("Options:")
        logger.info("  1. Make sure the model file exists at the specified path")
        logger.info("  2. Train the model using custom_dataset.py")
        logger.info("  3. Set USE_RESNET = False to disable ResNet")
        
        choice = input("Continue without ResNet? (y/n): ").lower()
        if choice == 'y':
            USE_RESNET = False
            resnet_model = None
            crowd_classifier = None
            logger.info("Continuing without ResNet model")
        else:
            logger.info("Please add your ResNet model and restart")
            exit(1)
    except Exception as e:
        logger.error(f"Error loading ResNet model: {e}")
        logger.info("Check if the model file is corrupted or has the wrong format")
        
        choice = input("Continue without ResNet? (y/n): ").lower()
        if choice == 'y':
            USE_RESNET = False
            resnet_model = None
            crowd_classifier = None
            logger.info("Continuing without ResNet model")
        else:
            logger.info("Please fix the model and restart")
            exit(1)

resnet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tracker = DeepSort(
    max_age=50,
    n_init=3,
    max_cosine_distance=0.4,
    nn_budget=None,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=device.type == 'mps',
    embedder_wts=None,
    polygon=False,
    today=None
)

def analyze_crowd_density(frame, model, classifier, transform, device):
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

def process_resnet_batch(crops, model, transform, device):
    if not crops or model is None:
        return []
    
    results = []
    
    for crop in crops:
        if crop.size > 0:
            try:
                tensor = transform(crop).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    density_output = model(tensor)
                    person_density = torch.sum(density_output).item()
                    
                    is_crowded = person_density > 0.5
                    confidence = min(person_density, 1.0)
                    
                    results.append((1 if is_crowded else 0, confidence))
            except:
                results.append((0, 0.0))
        else:
            results.append((0, 0.0))
    
    return results

def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        logger.error(f"Cannot access camera {CAMERA_ID}")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    logger.info("Press 'q' to quit.")
    logger.info(f"Using {'ResNet Crowd Density + YOLO + DeepSORT' if USE_RESNET else 'YOLO + DeepSORT'} on {device}")

    frame_count = 0
    csv_file = open(CSV_FILE, mode='a', newline='')
    csv_writer = csv.writer(csv_file)
    
    processing_times = []
    
    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera error, attempting to reconnect...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(CAMERA_ID)
                continue

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            raw_filename = f"cam{CAMERA_ID}_{timestamp}.jpg"
            
            crowd_count = 0.0
            crowd_level = "Unknown"
            crowd_confidence = 0.0
            
            if USE_RESNET:
                crowd_count, crowd_level, crowd_confidence = analyze_crowd_density(
                    frame, resnet_model, crowd_classifier, resnet_transform, device
                )

            people_boxes = []
            crowded_people_count = 0
            
            results = yolo_model(frame, verbose=False, conf=MIN_CONFIDENCE)
            
            crops_for_resnet = []
            crop_indices = []
            
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if yolo_model.names[cls] == "person":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                            continue
                            
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        label = "YOLO"
                        final_conf = conf

                        bbox_area = (x2 - x1) * (y2 - y1)
                        is_potentially_crowded = conf < 0.6 or bbox_area < 1500

                        if USE_RESNET and is_potentially_crowded:
                            crops_for_resnet.append(crop)
                            crop_indices.append(len(people_boxes))
                            people_boxes.append((x1, y1, x2, y2, label, final_conf, True))
                        else:
                            people_boxes.append((x1, y1, x2, y2, label, final_conf, False))

            if USE_RESNET and crops_for_resnet:
                resnet_results = process_resnet_batch(crops_for_resnet, resnet_model, resnet_transform, device)
                
                for i, (prediction, confidence) in enumerate(resnet_results):
                    if prediction == 1 and confidence > 0.3:
                        box_idx = crop_indices[i]
                        x1, y1, x2, y2, _, _, _ = people_boxes[box_idx]
                        people_boxes[box_idx] = (x1, y1, x2, y2, "Crowded Person", confidence, False)
                        crowded_people_count += 1

            people_boxes = [(x1, y1, x2, y2, label, conf) for x1, y1, x2, y2, label, conf, flag in people_boxes if not flag or USE_RESNET]

            formatted_detections = []
            for x1, y1, x2, y2, label, conf in people_boxes:
                formatted_detections.append(([x1, y1, x2, y2], conf, 0))

            tracks = tracker.update_tracks(formatted_detections, frame=frame)
            
            tracked_people = []
            active_track_ids = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                active_track_ids.append(track_id)
                
                best_match = None
                best_iou = 0
                for det in people_boxes:
                    det_box = det[:4]
                    iou = calculate_iou(ltrb, det_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = det
                
                if best_match and best_iou > 0.3:
                    label = f"{best_match[4]} (ID:{track_id})"
                    conf = best_match[5]
                else:
                    label = f"Tracked (ID:{track_id})"
                    conf = 0.7
                
                tracked_people.append((x1, y1, x2, y2, label, conf))

            people_count = len(tracked_people)
            density = people_count / CAMERA_FOV_M2
            device_status = "ON" if density > DENSITY_THRESHOLD else "OFF"

            try:
                if device_status == "ON":
                    turn_on_device()
                else:
                    turn_off_device()
            except Exception as e:
                logger.warning(f"GPIO control error: {e}")

            processing_time = (time.time() - start_time) * 1000
            processing_times.append(processing_time)
            
            tracked_ids_str = ",".join(map(str, active_track_ids)) if active_track_ids else "None"
            csv_writer.writerow([
                timestamp.replace('_', ' '), 
                f"Camera {CAMERA_ID}",
                people_count, 
                f"{density:.4f}", 
                device_status, 
                f"{crowd_level} ({crowd_confidence:.2f})" if USE_RESNET else "N/A",
                tracked_ids_str,
                f"{processing_time:.2f}"
            ])

            for (x1, y1, x2, y2, label, conf) in tracked_people:
                if "Crowded" in label:
                    color = (0, 0, 255)
                elif "Tracked" in label:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            avg_fps = 1000 / np.mean(processing_times[-30:]) if processing_times else 0
            cv2.putText(frame, f"People: {people_count} (Crowded: {crowded_people_count})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Density: {density:.4f} ppl/m² | Devices: {device_status}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Crowd Level: {crowd_level} ({crowd_confidence:.2f})" if USE_RESNET else "Crowd Level: N/A",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(frame, f"Tracks: {len(active_track_ids)} | FPS: {avg_fps:.1f} | Device: {device}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"ResNet: {'ON' if USE_RESNET else 'OFF'} | Frame: {frame_count}",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imwrite(os.path.join(OUTPUT_DIR, raw_filename), frame)
            cv2.imshow("People Density Monitoring", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_count % 100 == 0:
                if device.type == 'mps':
                    torch.mps.empty_cache()
                avg_proc_time = np.mean(processing_times[-100:])
                logger.info(f"Frame {frame_count}: Avg processing time: {avg_proc_time:.2f}ms, FPS: {1000/avg_proc_time:.1f}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        cleanup_gpio()
        logger.info("Cleanup completed")
        
        if processing_times:
            avg_time = np.mean(processing_times)
            logger.info(f"Average processing time: {avg_time:.2f}ms")
            logger.info(f"Average FPS: {1000/avg_time:.1f}")

if __name__ == "__main__":
    main()
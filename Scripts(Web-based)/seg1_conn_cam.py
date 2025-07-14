import cv2
import numpy as np
from datetime import datetime
import os
import time

# Create directory for saved photos if it doesn't exist
os.makedirs('detected_persons', exist_ok=True)

# Load YOLO model
def load_yolo():
    net = cv2.dnn.readNet("yolov3-spp.weights", "yolov3-spp.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Detect objects using YOLO
def detect_objects(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs

# Get detection information
def get_box_dimensions(outputs, height, width):
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 0 is person class in COCO
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids

# Track detected persons and capture new ones
class PersonTracker:
    def __init__(self):
        self.detected_persons = set()
        self.last_capture_time = 0
        self.capture_delay = 2  # seconds between captures
    
    def check_new_persons(self, boxes, frame):
        current_time = time.time()
        if current_time - self.last_capture_time < self.capture_delay:
            return False
        
        new_detections = set()
        for box in boxes:
            # Create a unique identifier for each person based on their position
            person_id = f"{box[0]}_{box[1]}"
            new_detections.add(person_id)
            
            if person_id not in self.detected_persons:
                self.detected_persons.add(person_id)
                self.last_capture_time = current_time
                return True
        return False

def main():
    # Load YOLO
    net, classes, output_layers = load_yolo()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    tracker = PersonTracker()
    photo_count = 0
    
    print("Person detection started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        height, width, channels = frame.shape
        
        # Detect objects
        outputs = detect_objects(frame, net, output_layers)
        boxes, confidences, class_ids = get_box_dimensions(outputs, height, width)
        
        # Draw bounding boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Check for new persons and capture
        if tracker.check_new_persons(boxes, frame):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detected_persons/person_{timestamp}_{photo_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"New person detected! Photo saved as {filename}")
            photo_count += 1
        
        # Display
        cv2.imshow("Person Detection", frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Detection ended")

if __name__ == "__main__":
    main()
import cv2
import sys

def test_camera_sources():
    """Test different camera sources to find working ones"""
    print("Testing camera sources...")
    
    # Test built-in camera indices
    for i in range(5):
        print(f"\n--- Testing Camera Index {i} ---")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Camera {i}: Working - Frame shape: {frame.shape}")
                # Test a few frames to ensure stability
                for j in range(3):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"⚠️  Camera {i}: Unstable - Failed on frame {j+2}")
                        break
                else:
                    print(f"✅ Camera {i}: Stable")
            else:
                print(f"❌ Camera {i}: Opened but no frames")
        else:
            print(f"❌ Camera {i}: Cannot open")
        
        cap.release()
    
    # Test common macOS camera paths
    macos_paths = [
        0,  # Default
        1,  # External camera
        "/dev/video0",
        "/dev/video1"
    ]
    
    print("\n--- Testing macOS Specific Paths ---")
    for path in macos_paths:
        print(f"Testing: {path}")
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ {path}: Working")
            else:
                print(f"⚠️  {path}: Opened but no frames")
        else:
            print(f"❌ {path}: Cannot open")
        cap.release()

def interactive_camera_test():
    """Interactive test with live preview"""
    print("\n" + "="*50)
    print("INTERACTIVE CAMERA TEST")
    print("="*50)
    
    camera_id = input("Enter camera ID to test (0, 1, 2, etc.) or 'q' to quit: ")
    
    if camera_id.lower() == 'q':
        return
    
    try:
        camera_id = int(camera_id)
    except ValueError:
        print("Invalid input. Using camera 0.")
        camera_id = 0
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_id}")
        return
    
    print(f"✅ Camera {camera_id} opened successfully!")
    print("Press 'q' to quit the preview, 's' to save a test frame")
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Add info overlay
            cv2.putText(frame, f"Camera {camera_id} - Frame {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow(f"Camera {camera_id} Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"test_frame_camera_{camera_id}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Saved frame to {filename}")
                
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"📊 Total frames processed: {frame_count}")

if __name__ == "__main__":
    print("🎥 Camera Detection and Testing Tool")
    print("="*40)
    
    # Automatic detection
    test_camera_sources()
    
    # Interactive testing
    while True:
        interactive_camera_test()
        
        continue_test = input("\nTest another camera? (y/n): ").lower()
        if continue_test != 'y':
            break
    
    print("👋 Camera testing completed!")
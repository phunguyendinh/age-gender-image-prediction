import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
import time
import argparse
from model import AgeGenderClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def load_model(model_path='saved_models/best_model.pth'):
    """Load trained model"""
    model = models.vgg16(pretrained=False)  
    
    model.avgpool = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )
    
    model.classifier = AgeGenderClassifier()
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from {model_path}")
    print(f"Model was trained for {checkpoint['epoch']+1} epochs")
    print(f"Best validation accuracy: {checkpoint['val_gender_acc']*100:.2f}%")
    print(f"Best validation age MAE: {checkpoint['val_age_mae']:.2f}")
    
    return model

def list_cameras():
    """Liệt kê tất cả camera có sẵn"""
    print("Scanning for available cameras...")
    available_cameras = []
    
    # Test camera từ index 0 đến 10
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                # Lấy thông tin camera
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                # Thử lấy tên camera (không phải tất cả driver đều hỗ trợ)
                camera_name = f"Camera {i}"
                
                camera_info = {
                    'index': i,
                    'name': camera_name,
                    'resolution': f"{width}x{height}",
                    'fps': fps,
                    'available': True
                }
                available_cameras.append(camera_info)
                print(f"  Camera {i}: {width}x{height} @ {fps}fps")
            cap.release()
        else:
            continue
    
    if not available_cameras:
        print(" No cameras found!")
        return []
    
    print(f"  Found {len(available_cameras)} camera(s)")
    return available_cameras

def select_camera():
    """Cho phép user chọn camera"""
    cameras = list_cameras()
    
    if not cameras:
        print("No cameras available!")
        return None
    
    if len(cameras) == 1:
        print(f"Using only available camera: Camera {cameras[0]['index']}")
        return cameras[0]['index']
    
    print("Available cameras:")
    for i, cam in enumerate(cameras):
        print(f"  {i+1}. Camera {cam['index']}: {cam['resolution']} @ {cam['fps']}fps")
    
    while True:
        try:
            choice = input(f"\nSelect camera (1-{len(cameras)}) or press Enter for default (Camera 0): ").strip()
            
            if choice == "":
                return 0  # Default camera
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(cameras):
                selected_camera = cameras[choice_idx]['index']
                print(f"Selected Camera {selected_camera}")
                return selected_camera
            else:
                print(f"Please enter a number between 1 and {len(cameras)}")
        
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("Exiting...")
            return None

def test_camera_settings(camera_index):
    """Test và hiển thị settings của camera"""
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return None
    
    # Lấy thông tin hiện tại
    original_settings = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
        'contrast': cap.get(cv2.CAP_PROP_CONTRAST),
        'saturation': cap.get(cv2.CAP_PROP_SATURATION),
    }
    
    print(f"Camera {camera_index} current settings:")
    print(f"  Resolution: {original_settings['width']}x{original_settings['height']}")
    print(f"  FPS: {original_settings['fps']}")
    print(f"  Brightness: {original_settings['brightness']:.2f}")
    print(f"  Contrast: {original_settings['contrast']:.2f}")
    print(f"  Saturation: {original_settings['saturation']:.2f}")
    
    # Thử optimize settings
    print(f"Optimizing camera settings...")
    
    # Set optimal resolution và FPS cho age-gender detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    # Verify settings
    new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"  New resolution: {new_width}x{new_height}")
    print(f"  New FPS: {new_fps}")
    
    return cap

def preprocess_image(image):
    """Preprocess image for model input"""
    image = cv2.resize(image, (224, 224))
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_tensor = torch.tensor(image).permute(2, 0, 1).float()
    image_tensor = normalize(image_tensor / 255.)
    
    return image_tensor.unsqueeze(0).to(device)

def predict_age_gender(model, image):
    """Predict age and gender from image"""
    with torch.no_grad():
        processed_image = preprocess_image(image)
        gender_pred, age_pred = model(processed_image)
        
        gender_prob = gender_pred.cpu().numpy()[0][0]
        age_normalized = age_pred.cpu().numpy()[0][0]
        
        predicted_age = int(age_normalized * 80)
        predicted_gender = "Female" if gender_prob > 0.5 else "Male"
        gender_confidence = gender_prob if gender_prob > 0.5 else 1 - gender_prob
        
        return predicted_age, predicted_gender, gender_confidence * 100

def detect_faces(image):
    """Detect faces using OpenCV Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
    
    return faces

def main(camera_index=None, model_path='saved_models/best_model.pth'):
    """Main function to run camera test"""
    print("Age-Gender Detection Camera Test")
    print("="*50)
    
    # Load model
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        print("Model file not found! Please make sure you have trained the model first.")
        return
    
    # Select camera
    if camera_index is None:
        camera_index = select_camera()
        if camera_index is None:
            return
    
    # Test and setup camera
    cap = test_camera_settings(camera_index)
    if cap is None:
        return
    
    print(f"Starting camera test with Camera {camera_index}")
    print("Controls:")
    print("  'q' = Quit")
    print("  's' = Save screenshot") 
    print("  'r' = Reset/Clear")
    print("  'c' = Switch camera")
    print("  'i' = Show camera info")
    print("  'space' = Pause/Resume")
    
    # Performance tracking
    fps_counter = 0
    start_time = time.time()
    total_faces_detected = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            display_frame = frame.copy()
            
            # Detect faces
            faces = detect_faces(frame)
            total_faces_detected += len(faces)
            
            # Process each face
            for i, (x, y, w, h) in enumerate(faces):
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size > 0: 
                    try:
                        age, gender, confidence = predict_age_gender(model, face_img)
                        
                        # Choose color based on confidence
                        color = (0, 255, 0) if confidence > 70 else (0, 255, 255)
                        
                        # Draw bounding box
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                        
                        # Draw face ID
                        cv2.putText(display_frame, f"ID:{i+1}", (x, y-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Prepare text
                        text1 = f"Age: {age}"
                        text2 = f"{gender} ({confidence:.1f}%)"
                        
                        # Draw text background
                        cv2.rectangle(display_frame, (x, y - 60), (x + 250, y), (0, 0, 0), -1)
                        
                        # Draw text
                        cv2.putText(display_frame, text1, (x + 5, y - 35), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(display_frame, text2, (x + 5, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                    except Exception as e:
                        print(f"Error in prediction: {e}")
                        continue
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:  
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = end_time
        
        else:
            # Use last frame when paused
            cv2.putText(display_frame, "PAUSED - Press SPACE to resume", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw info overlay
        info_y = 30
        cv2.putText(display_frame, f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: Calculating...", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(display_frame, f"Camera: {camera_index} | Faces: {len(faces)} | Total: {total_faces_detected}", 
                   (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Status
        status = "PAUSED" if paused else "RUNNING"
        cv2.putText(display_frame, f"Status: {status}", 
                   (10, display_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(display_frame, "Q=Quit | S=Save | C=Camera | Space=Pause", 
                   (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow(f'Age-Gender Detection - Camera {camera_index}', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = int(time.time())
            filename = f"camera_{camera_index}_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"Screenshot saved as {filename}")
        elif key == ord('r'):
            print("Resetting...")
            total_faces_detected = 0
            fps_counter = 0
            start_time = time.time()
        elif key == ord('c'):
            print("Switching camera...")
            cap.release()
            new_camera = select_camera()
            if new_camera is not None:
                camera_index = new_camera
                cap = test_camera_settings(camera_index)
                if cap is None:
                    print("Failed to switch camera")
                    break
        elif key == ord('i'):
            print(f"Camera {camera_index} Info:")
            print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
            print(f"  Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
            print(f"  Total faces detected: {total_faces_detected}")
        elif key == ord(' '):  # Space key
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test completed!")
    
    # Final stats
    print(f"  Session Statistics:")
    print(f"  Total faces detected: {total_faces_detected}")
    print(f"  Final FPS: {fps:.1f}" if 'fps' in locals() else "  FPS: Not calculated")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Age-Gender Detection Camera Test')
    parser.add_argument('-c', '--camera', type=int, help='Camera index (0, 1, 2, ...)')
    parser.add_argument('-m', '--model', default='saved_models/best_model.pth',
                       help='Model path (default: saved_models/best_model.pth)')
    parser.add_argument('-l', '--list', action='store_true',
                       help='List available cameras and exit')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.list:
        list_cameras()
    else:
        main(camera_index=args.camera, model_path=args.model)
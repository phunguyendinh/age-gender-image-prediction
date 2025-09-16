import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
import time
import os
import argparse
from pathlib import Path
from model import AgeGenderClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

class VideoProcessor:
    """Xử lý video với age-gender prediction"""
    def __init__(self, model_path='saved_models/best_model.pth'):
        self.device = device
        self.model = self.load_model(model_path)
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.input_video = None
        self.output_video = None
        self.total_frames = 0
        self.processed_frames = 0
        
        self.fps_list = []
        self.start_time = 0
    
    def load_model(self, model_path):
        """Load trained model"""
        model = models.vgg16(pretrained=False)
        model.avgpool = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten()
        )
        model.classifier = AgeGenderClassifier()
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            print(f"Model loaded successfully from {model_path}")
            
            if 'val_gender_acc' in checkpoint:
                print(f"Model accuracy: {checkpoint['val_gender_acc']*100:.2f}%")
            if 'val_age_mae' in checkpoint:
                print(f"Age MAE: {checkpoint['val_age_mae']:.2f}")
                
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image).permute(2, 0, 1).float()
        image_tensor = self.normalize(image_tensor / 255.)
        return image_tensor.unsqueeze(0).to(self.device)
    
    def predict_age_gender(self, image):
        """Predict age and gender"""
        if self.model is None:
            return None
        
        try:
            processed_image = self.preprocess_image(image)
            
            with torch.no_grad():
                gender_pred, age_pred = self.model(processed_image)
                
                gender_prob = gender_pred.cpu().numpy()[0][0]
                age_normalized = age_pred.cpu().numpy()[0][0]
                
                predicted_age = int(age_normalized * 80)
                predicted_gender = "Female" if gender_prob > 0.5 else "Male"
                confidence = gender_prob if gender_prob > 0.5 else 1 - gender_prob
                
                return {
                    'age': predicted_age,
                    'gender': predicted_gender,
                    'confidence': confidence * 100,
                    'raw_gender_prob': gender_prob
                }
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
    
    def draw_predictions(self, frame, faces, predictions):
        """Draw bounding boxes and predictions on frame"""
        for i, ((x, y, w, h), pred) in enumerate(zip(faces, predictions)):
            if pred is None:
                continue
            
            color = (0, 255, 0) if pred['confidence'] > 70 else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            text1 = f"Age: {pred['age']}"
            text2 = f"{pred['gender']} ({pred['confidence']:.1f}%)"
            text3 = f"ID: {i+1}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (w1, h1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
            (w2, h2), _ = cv2.getTextSize(text2, font, font_scale, thickness)
            (w3, h3), _ = cv2.getTextSize(text3, font, font_scale, thickness)
            
            max_width = max(w1, w2, w3) + 10
            total_height = h1 + h2 + h3 + 30
            
            cv2.rectangle(frame, (x, y - total_height), (x + max_width, y), (0, 0, 0), -1)
            cv2.rectangle(frame, (x, y - total_height), (x + max_width, y), color, 2)
            
            cv2.putText(frame, text3, (x + 5, y - total_height + h3 + 5), 
                       font, font_scale, (255, 255, 255), thickness)
            cv2.putText(frame, text1, (x + 5, y - total_height + h3 + h1 + 15), 
                       font, font_scale, (255, 255, 255), thickness)
            cv2.putText(frame, text2, (x + 5, y - total_height + h3 + h1 + h2 + 25), 
                       font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def process_video(self, input_path, output_path=None, display=True, skip_frames=0):
        """Process entire video file"""
        self.input_video = cv2.VideoCapture(input_path)
        if not self.input_video.isOpened():
            print(f"✗ Cannot open video: {input_path}")
            return False
        
        fps = int(self.input_video.get(cv2.CAP_PROP_FPS))
        width = int(self.input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {self.total_frames} frames")
        print(f"Duration: {self.total_frames/fps:.2f} seconds")
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        frame_count = 0
        face_count_total = 0
        prediction_times = []
        self.start_time = time.time()
        
        print(f"Starting video processing...")
        print("Press 'q' to quit, 's' to save current frame, 'space' to pause")
        
        while True:
            ret, frame = self.input_video.read()
            if not ret:
                break
            
            frame_count += 1
            self.processed_frames = frame_count
            
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                continue
            
            frame_start_time = time.time()
            
            faces = self.detect_faces(frame)
            face_count_total += len(faces)
            
            predictions = []
            for (x, y, w, h) in faces:
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(width, x + w + padding)
                y2 = min(height, y + h + padding)
                
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size > 0:
                    pred = self.predict_age_gender(face_img)
                    predictions.append(pred)
                else:
                    predictions.append(None)
            
            frame_with_predictions = self.draw_predictions(frame.copy(), faces, predictions)
            
            progress = (frame_count / self.total_frames) * 100
            elapsed_time = time.time() - self.start_time
            estimated_total = (elapsed_time / frame_count) * self.total_frames
            remaining_time = estimated_total - elapsed_time
            
            info_text = [
                f"Frame: {frame_count}/{self.total_frames} ({progress:.1f}%)",
                f"Faces detected: {len(faces)} (Total: {face_count_total})",
                f"Time: {elapsed_time:.1f}s / {estimated_total:.1f}s",
                f"Remaining: {remaining_time:.1f}s",
                f"Processing FPS: {frame_count/elapsed_time:.1f}"
            ]
            
            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(frame_with_predictions, text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(frame_with_predictions, "Q=Quit, S=Save, Space=Pause", 
                       (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            frame_processing_time = time.time() - frame_start_time
            prediction_times.append(frame_processing_time)
            
            if self.output_video:
                self.output_video.write(frame_with_predictions)
            
            if display:
                cv2.imshow('Video Age-Gender Detection', frame_with_predictions)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Processing stopped by user")
                    break
                elif key == ord('s'):
                    save_name = f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(save_name, frame_with_predictions)
                    print(f"Saved frame: {save_name}")
                elif key == ord(' '):
                    print("Paused - Press any key to continue")
                    cv2.waitKey(0)
            
            if frame_count % 100 == 0:
                avg_time = np.mean(prediction_times[-100:])
                print(f"Progress: {frame_count}/{self.total_frames} "
                      f"({progress:.1f}%) - Avg processing time: {avg_time:.3f}s/frame")
        
        self.cleanup()
        self.print_statistics(frame_count, face_count_total, prediction_times)
        
        return True
    
    def cleanup(self):
        """Clean up resources"""
        if self.input_video:
            self.input_video.release()
        if self.output_video:
            self.output_video.release()
        cv2.destroyAllWindows()
    
    def print_statistics(self, total_frames, total_faces, processing_times):
        """Print processing statistics"""
        total_time = time.time() - self.start_time
        avg_fps = total_frames / total_time
        avg_processing_time = np.mean(processing_times)
        
        print(f"Processing Statistics:")
        print(f"Total frames processed: {total_frames}")
        print(f"Total faces detected: {total_faces}")
        print(f"Average faces per frame: {total_faces/total_frames:.2f}")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average processing time per frame: {avg_processing_time:.3f}s")
        print(f"{'Real-time capable' if avg_processing_time < 0.033 else '⚠️ Slower than real-time'}")

def main():
    parser = argparse.ArgumentParser(description='Age-Gender Detection on Video')
    parser.add_argument('input', help='Input video path')
    parser.add_argument('-o', '--output', help='Output video path (optional)')
    parser.add_argument('-m', '--model', default='saved_models/best_model.pth',
                       help='Model path (default: saved_models/best_model.pth)')
    parser.add_argument('--no-display', action='store_true',
                       help='Process without displaying video')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Skip frames for faster processing (0 = no skip)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return
    
    if args.output and os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    
    processor = VideoProcessor(args.model)
    success = processor.process_video(
        input_path=args.input,
        output_path=args.output,
        display=not args.no_display,
        skip_frames=args.skip_frames
    )
    
    if success:
        print("Video processing completed successfully!")
    else:
        print("Video processing failed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        print("Video Test Mode")
        print("Usage examples:")
        print("python video_test.py input_video.mp4")
        print("python video_test.py input_video.mp4 -o output_video.mp4")
        print("python video_test.py input_video.mp4 --skip-frames 2")
        print()
        
        test_videos = ['video.mp4']
        found_video = None
        
        for video in test_videos:
            if os.path.exists(video):
                found_video = video
                break
        
        if found_video:
            print(f"Found test video: {found_video}")
            processor = VideoProcessor()
            processor.process_video(found_video, display=True)
        else:
            print("No test video found. Please provide video path as argument.")
            print("Example: python video_test.py your_video.mp4")
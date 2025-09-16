import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from model import AgeGenderClassifier

def create_model(pretrained=True):
    """Create the age-gender prediction model"""
    model = models.vgg16(pretrained=pretrained)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.avgpool = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )
    
    model.classifier = AgeGenderClassifier()
    
    return model

def load_trained_model(model_path, device='cuda'):
    """Load a trained model from checkpoint"""
    model = create_model(pretrained=False)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        if 'epoch' in checkpoint:
            print(f"Trained for {checkpoint['epoch']+1} epochs")
        if 'val_gender_acc' in checkpoint:
            print(f"Best gender accuracy: {checkpoint['val_gender_acc']*100:.2f}%")
        if 'val_age_mae' in checkpoint:
            print(f"Best age MAE: {checkpoint['val_age_mae']:.2f}")
            
        return model, checkpoint
        
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def preprocess_single_image(image_path_or_array, target_size=224):
    """Preprocess a single image for prediction"""
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path_or_array
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, (target_size, target_size))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    image_tensor = torch.tensor(image).permute(2, 0, 1).float()
    image_tensor = normalize(image_tensor / 255.)
    
    return image_tensor.unsqueeze(0) 

def predict_age_gender(model, image, device='cuda'):
    """Make prediction on a single image"""
    model.eval()
    
    image_tensor = preprocess_single_image(image).to(device)
    
    with torch.no_grad():
        gender_pred, age_pred = model(image_tensor)
        
        gender_prob = gender_pred.cpu().numpy()[0][0]
        age_normalized = age_pred.cpu().numpy()[0][0]
        
        predicted_age = int(age_normalized * 80) 
        predicted_gender = "Female" if gender_prob > 0.5 else "Male"
        gender_confidence = gender_prob if gender_prob > 0.5 else 1 - gender_prob
        
        return {
            'age': predicted_age,
            'gender': predicted_gender,
            'gender_probability': gender_prob,
            'gender_confidence': gender_confidence * 100,
            'raw_age': age_normalized,
            'raw_gender': gender_prob
        }

def test_on_image(model_path, image_path, device='cuda'):
    """Test model on a single image and display result"""
    model, checkpoint = load_trained_model(model_path, device)
    if model is None:
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    result = predict_age_gender(model, image, device)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title(f"Predicted: {result['gender']}, Age: {result['age']}\n"
              f"Confidence: {result['gender_confidence']:.1f}%")
    plt.axis('off')
    plt.show()
    
    print(f"Prediction Results:")
    print(f"  Gender: {result['gender']} (confidence: {result['gender_confidence']:.1f}%)")
    print(f"  Age: {result['age']} years")
    
    return result

def batch_predict(model, image_list, device='cuda'):
    """Make predictions on a batch of images"""
    model.eval()
    results = []
    
    for image in image_list:
        result = predict_age_gender(model, image, device)
        results.append(result)
    
    return results

def evaluate_model_performance(model_path, test_images, test_labels, device='cuda'):
    """Evaluate model performance on test set"""
    model, _ = load_trained_model(model_path, device)
    if model is None:
        return
    
    model.eval()
    correct_gender = 0
    total_age_error = 0
    total_samples = len(test_images)
    
    with torch.no_grad():
        for i, (image_path, true_age, true_gender) in enumerate(zip(test_images, test_labels['age'], test_labels['gender'])):
            try:
                result = predict_age_gender(model, image_path, device)
                
                if result['gender'].lower() == true_gender.lower():
                    correct_gender += 1
                
                age_error = abs(result['age'] - true_age)
                total_age_error += age_error
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{total_samples} images")
                    
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                continue
    
    gender_accuracy = correct_gender / total_samples * 100
    age_mae = total_age_error / total_samples
# Age-Gender Prediction System

A real-time age and gender prediction system using deep learning with PyTorch. The system can process live camera feeds and video files to detect faces and predict age and gender with confidence scores.

![Demo](video_preview/video.mp4)

## Features

- **Real-time Camera Processing**: Live age and gender prediction from camera feeds
- **Video File Processing**: Batch processing of video files with output saving
- **Face Detection**: Automatic face detection using OpenCV Haar Cascades
- **Multi-Camera Support**: Automatic detection and selection of available cameras
- **GPU Acceleration**: CUDA support for faster inference
- **Interactive Controls**: Runtime controls for pause, screenshot, camera switching
- **Performance Monitoring**: FPS tracking and processing statistics

## Architecture

The system uses a VGG16-based architecture with custom classifier heads:
- **Backbone**: VGG16 (without pre-trained weights)
- **Feature Extractor**: Custom average pooling layer with Conv2D
- **Classifier**: Dual-head network for age and gender prediction
- **Age Output**: Normalized value (0-1) representing age 0-80
- **Gender Output**: Binary classification (Male/Female) with confidence

## Requirements

```
torch
torchvision
opencv-python
numpy
```

## Installation

1. Clone the repository
2. Install required dependencies:
```bash
pip install torch torchvision opencv-python numpy
```
3. Ensure you have a trained model file in `saved_models/best_model.pth`

## Usage

### Camera Testing

**List available cameras:**
```bash
python camera_test.py -l
```

**Use specific camera:**
```bash
python camera_test.py -c 1
```

**Use custom model:**
```bash
python camera_test.py -c 0 -m path/to/model.pth
```

**Interactive camera selection:**
```bash
python camera_test.py
```

#### Camera Controls
- `q` - Quit application
- `s` - Save screenshot
- `r` - Reset statistics
- `c` - Switch camera
- `i` - Show camera information
- `space` - Pause/Resume

### Video Processing

**Basic video processing (display only):**
```bash
python video_test.py input_video.mp4
```

**Process and save output:**
```bash
python video_test.py input_video.mp4 -o output.mp4
```

**Skip frames for faster processing:**
```bash
python video_test.py input_video.mp4 --skip-frames 2
```

**Process without display (headless):**
```bash
python video_test.py input_video.mp4 --no-display
```

#### Video Controls
- `q` - Quit processing
- `s` - Save current frame
- `space` - Pause/Resume

## Model Structure

The model consists of three main components:

1. **Feature Extraction**: Modified VGG16 with custom average pooling
2. **Intermediate Layers**: Fully connected layers with dropout for regularization
3. **Output Heads**: Separate classifiers for age and gender prediction

```python
# Age output: Sigmoid activation (0-1, representing 0-80 years)
# Gender output: Sigmoid activation (>0.5 = Female, <0.5 = Male)
```

## Performance Features

- **Face Detection**: OpenCV Haar Cascade for robust face detection
- **Batch Processing**: Efficient video processing with progress tracking
- **Resource Management**: Automatic cleanup and memory management
- **Error Handling**: Comprehensive error handling for various edge cases
- **Statistics**: Detailed processing statistics and performance metrics

## Output Information

For each detected face, the system provides:
- **Age**: Predicted age (0-80 years)
- **Gender**: Male/Female classification
- **Confidence**: Prediction confidence percentage
- **Bounding Box**: Face location with color-coded confidence
  - Green: High confidence (>70%)
  - Yellow: Medium confidence (≤70%)

## Command Line Arguments

### Camera Test
```
-c, --camera    Camera index (0, 1, 2, ...)
-m, --model     Model path (default: saved_models/best_model.pth)
-l, --list      List available cameras and exit
```

### Video Test
```
input           Input video path
-o, --output    Output video path (optional)
-m, --model     Model path (default: saved_models/best_model.pth)
--no-display    Process without displaying video
--skip-frames   Skip frames for faster processing (0 = no skip)
```

## Technical Details

- **Input Resolution**: 224x224 pixels for model input
- **Preprocessing**: ImageNet normalization
- **Face Padding**: 20-pixel padding around detected faces
- **Minimum Face Size**: 30x30 pixels (video) / 50x50 pixels (camera)
- **Color Space**: BGR to RGB conversion for model compatibility

## Troubleshooting

**Model not found error:**
- Ensure the model file exists in `saved_models/best_model.pth`
- Check the model path if using custom location

**Camera not accessible:**
- Check camera permissions
- Try different camera indices
- Ensure camera is not being used by other applications

**Low FPS performance:**
- Use GPU acceleration if available
- Reduce input resolution
- Use frame skipping for video processing

**No faces detected:**
- Ensure adequate lighting
- Check camera positioning
- Verify face is clearly visible and not too small

## File Structure

```
├── camera_test.py          # Real-time camera processing
├── video_test.py           # Video file processing
├── model.py                # Model architecture definition
├── saved_models/           # Trained model storage
│   └── best_model.pth     # Pre-trained model weights
└── README.md              # This file
```

## Model Information

When loading a model, the system displays:
- Training epochs completed
- Validation accuracy for gender prediction
- Mean Absolute Error (MAE) for age prediction
- Device being used (CPU/CUDA)

This information helps verify model quality and expected performance.
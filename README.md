# Face Recognition System

A comprehensive face recognition system that combines **RetinaFace** for face detection, **InsightFace** (ArcFace + iResNet100) for face recognition, and **FAISS** for efficient similarity search and storage.

## Features

- **Face Detection**: Uses RetinaFace for accurate face detection with landmarks
- **Face Recognition**: Leverages InsightFace's ArcFace + iResNet100 for high-accuracy face recognition
- **Efficient Storage**: FAISS index for fast similarity search and storage
- **Database Management**: Add, remove, and manage face entries
- **Persistence**: Save and load face databases
- **Batch Processing**: Process multiple images efficiently

## Requirements

- Python 3.7+
- CUDA-compatible GPU (recommended for better performance)
- See `requirements.txt` for all dependencies

## Installation

1. **Clone or download the project files**

2. **Install dependencies** (choose one method):

   **Method 1: Automatic installation script**
   ```bash
   python install_dependencies.py
   ```

   **Method 2: Manual installation**
   ```bash
   pip install -r requirements.txt
   ```

   **Method 3: Alternative installation (if Method 2 fails)**
   ```bash
   pip install -r requirements_alternative.txt
   pip install insightface --no-deps
   pip install retina-face
   ```

3. **Download InsightFace models** (automatically handled on first run):
   The system will automatically download the required InsightFace models on first initialization.

### Troubleshooting Installation Issues

If you encounter compilation errors (especially on Windows):

1. **Install Microsoft Visual C++ Build Tools**:
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install the "C++ build tools" workload

2. **Use Conda instead of pip**:
   ```bash
   conda install -c conda-forge insightface
   conda install -c conda-forge faiss-cpu
   ```

3. **Install pre-built wheels**:
   ```bash
   pip install --only-binary=all insightface
   ```

4. **Try the installation script**:
   ```bash
   python install_dependencies.py
   ```

## Quick Start

### Basic Usage

```python
from face_recognition_system import FaceRecognitionSystem, load_image

# Initialize the system
system = FaceRecognitionSystem()

# Load an image
image = load_image("path/to/face_image.jpg")

# Add a face to the database
face_id = system.add_face_to_database(image, "John Doe", {"age": 30})

# Recognize faces in a new image
test_image = load_image("path/to/test_image.jpg")
results = system.recognize_face(test_image, threshold=0.6)

for result in results:
    print(f"Found: {result['name']} (confidence: {result['confidence']:.3f})")

# Save the database
system.save_database("my_face_database")
```

### Running the Example

```bash
python example_usage.py
```

## System Architecture

### Components

1. **Face Detection (RetinaFace)**
   - Detects faces in images
   - Provides bounding boxes and facial landmarks
   - High accuracy even with challenging conditions

2. **Face Recognition (InsightFace)**
   - Uses ArcFace + iResNet100 architecture
   - Extracts 512-dimensional face embeddings
   - State-of-the-art recognition accuracy

3. **Storage & Search (FAISS)**
   - Efficient similarity search using cosine similarity
   - Fast retrieval even with large databases
   - Supports both CPU and GPU implementations

### Data Flow

```
Input Image → RetinaFace Detection → InsightFace Embedding → FAISS Search → Recognition Result
```

## API Reference

### FaceRecognitionSystem

#### Initialization
```python
system = FaceRecognitionSystem(model_path="models", embedding_dim=512)
```

#### Methods

**Face Detection**
```python
faces = system.detect_faces(image)
```
Returns list of detected faces with bounding boxes and landmarks.

**Add Face to Database**
```python
face_id = system.add_face_to_database(image, person_name, additional_info=None)
```
Adds a face to the database and returns the assigned ID.

**Face Recognition**
```python
results = system.recognize_face(image, threshold=0.6)
```
Recognizes faces in an image and returns results with confidence scores.

**Database Management**
```python
# Save database
system.save_database("database_name")

# Load database
system.load_database("database_name")

# Get statistics
stats = system.get_database_stats()

# Remove face
system.remove_face(face_id)
```

## Configuration

### Model Parameters

- **Detection Size**: 640x640 (configurable in `_initialize_models`)
- **Embedding Dimension**: 512 (InsightFace default)
- **Similarity Threshold**: 0.6 (recommended, adjustable per recognition call)

### Performance Tuning

- **GPU Usage**: Set `ctx_id=0` for GPU, `ctx_id=-1` for CPU
- **Detection Size**: Smaller sizes for faster processing, larger for better accuracy
- **FAISS Index**: Use `IndexFlatIP` for cosine similarity (current) or `IndexIVFFlat` for larger datasets

## File Structure

```
face_recognition/
├── face_recognition_system.py    # Main system implementation
├── example_usage.py              # Example usage script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── models/                       # Model storage (auto-created)
├── sample_images/                # Sample images (auto-created)
└── face_database_*               # Database files (auto-created)
```

## Usage Examples

### Adding Multiple Faces

```python
import os
from face_recognition_system import FaceRecognitionSystem, load_image

system = FaceRecognitionSystem()

# Add all faces from a directory
image_dir = "face_images"
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(image_dir, filename)
        image = load_image(image_path)
        person_name = os.path.splitext(filename)[0]
        
        try:
            face_id = system.add_face_to_database(image, person_name)
            print(f"Added {person_name} with ID {face_id}")
        except Exception as e:
            print(f"Failed to add {person_name}: {e}")
```

### Real-time Recognition

```python
import cv2
from face_recognition_system import FaceRecognitionSystem

system = FaceRecognitionSystem()
# Load existing database
system.load_database("my_database")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Recognize faces
    results = system.recognize_face(frame_rgb, threshold=0.6)
    
    # Draw results on frame
    for result in results:
        if result['bbox']:
            x1, y1, x2, y2 = map(int, result['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{result['name']} ({result['confidence']:.2f})", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce detection size in `_initialize_models`
   - Use CPU mode: `ctx_id=-1`

2. **No Faces Detected**
   - Check image quality and lighting
   - Ensure face is clearly visible and not too small
   - Try adjusting detection parameters

3. **Low Recognition Accuracy**
   - Increase similarity threshold
   - Add more face samples per person
   - Ensure good image quality and varied angles

4. **Model Download Issues**
   - Check internet connection
   - Manual download from InsightFace repository
   - Clear model cache and retry

### Performance Tips

- Use GPU for faster processing
- Batch process multiple images
- Use appropriate image sizes (640x640 recommended)
- Regular database maintenance (remove duplicates)

## License

This project uses open-source libraries:
- InsightFace: Apache 2.0
- RetinaFace: MIT
- FAISS: MIT
- OpenCV: BSD

## Contributing

Feel free to submit issues and enhancement requests!

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for the face recognition models
- [RetinaFace](https://github.com/deepinsight/insightface/tree/master/python-package) for face detection
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search 
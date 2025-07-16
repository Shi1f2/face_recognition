from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import cv2
import os
import numpy as np
from face_recognition_system import FaceRecognitionSystem, load_image

app = FastAPI()

class FaceToAdd(BaseModel):
    image_path: str
    name: str

class RecognizeVideoRequest(BaseModel):
    videos_path: str  # Path to the video file
    threshold: float = 0.6
    frame_interval: int = 30  # Sample every Nth frame
    max_frames: int = 100  # Max frames to process
    faces_to_add: List[FaceToAdd] = []

class FrameRecognitionResult(BaseModel):
    frame_number: int
    faces: List[Dict[str, Any]]

class RecognizeVideoResponse(BaseModel):
    video_path: str
    processed_frames: int
    results: List[FrameRecognitionResult]

# Initialize the face recognition system (load database if needed)
face_system = FaceRecognitionSystem()
# Optionally, load database here if you want persistent faces
# face_system.load_database('face_database')

@app.post("/recognize_video", response_model=RecognizeVideoResponse)
def recognize_video(req: RecognizeVideoRequest):
    # Add faces to the database before processing the video
    for face in req.faces_to_add:
        try:
            img = load_image(face.image_path)
            face_system.add_face_to_database(img, face.name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to add face {face.name}: {e}")

    video_path = req.videos_path
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Failed to open video file.")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []
    processed = 0
    frame_idx = 0
    
    while cap.isOpened() and processed < req.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % req.frame_interval == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_system.recognize_face(rgb_frame, threshold=req.threshold)
            results.append(FrameRecognitionResult(frame_number=frame_idx, faces=faces))
            processed += 1
        frame_idx += 1
    cap.release()
    
    return RecognizeVideoResponse(
        video_path=video_path,
        processed_frames=processed,
        results=results
    )
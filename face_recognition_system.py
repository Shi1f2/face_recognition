import cv2
import numpy as np
import faiss
import os
import pickle
from typing import List, Tuple, Optional, Dict
import insightface
from insightface.app import FaceAnalysis
from retinaface import RetinaFace
import onnxruntime as ort
from PIL import Image
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    """
    Face Recognition System using RetinaFace + InsightFace + FAISS
    """
    
    def __init__(self, model_path: str = "models", embedding_dim: int = 512):
        """
        Initialize the face recognition system
        
        Args:
            model_path: Path to store/load models
            embedding_dim: Dimension of face embeddings
        """
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        self.faiss_index = None
        self.face_database = {}  # id -> metadata mapping
        self.next_id = 0
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize face detection and recognition models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize RetinaFace and InsightFace models"""
        try:
            # Initialize InsightFace for face recognition
            self.face_analyzer = FaceAnalysis(name='buffalo_l')
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace model initialized successfully")
            
            # RetinaFace will be used through InsightFace's detection
            logger.info("Face detection and recognition models ready")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            # Try alternative initialization if the first fails
            try:
                logger.info("Trying alternative model initialization...")
                self.face_analyzer = FaceAnalysis(name='buffalo_l')
                self.face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))  # Use CPU
                logger.info("InsightFace model initialized successfully (CPU mode)")
            except Exception as e2:
                logger.error(f"Alternative initialization also failed: {e2}")
                raise
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image using RetinaFace
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected faces with bounding boxes and landmarks
        """
        try:
            # Use InsightFace's detection (which uses RetinaFace internally)
            faces = self.face_analyzer.get(image)
            return faces
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def extract_embeddings(self, image: np.ndarray, faces: List) -> List[np.ndarray]:
        """
        Extract face embeddings using InsightFace
        
        Args:
            image: Input image
            faces: List of detected faces
            
        Returns:
            List of face embeddings
        """
        embeddings = []
        for face in faces:
            try:
                # Get embedding from the face
                embedding = getattr(face, 'embedding', None)
                if embedding is not None:
                    embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error extracting embedding: {e}")
                continue
        return embeddings
    
    def add_face_to_database(self, image: np.ndarray, person_name: str, 
                           additional_info: Optional[Dict] = None) -> int:
        """
        Add a face to the database
        
        Args:
            image: Input image
            person_name: Name of the person
            additional_info: Additional metadata
            
        Returns:
            Face ID assigned to the person
        """
        # Detect faces
        faces = self.detect_faces(image)
        
        if not faces:
            raise ValueError("No faces detected in the image")
        
        if len(faces) > 1:
            logger.warning(f"Multiple faces detected, using the first one")
        
        # Extract embedding
        embeddings = self.extract_embeddings(image, faces)
        
        if not embeddings:
            raise ValueError("Failed to extract face embedding")
        
        embedding = embeddings[0]
        
        # Add to FAISS index
        if self.faiss_index is None:
            # Initialize FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Normalize embedding for cosine similarity
        embedding_normalized = embedding / np.linalg.norm(embedding)
        
        # Add to index
        self.faiss_index.add(embedding_normalized.reshape(1, -1))  # type: ignore
        
        # Store metadata
        face_id = self.next_id
        self.face_database[face_id] = {
            'name': person_name,
            'embedding': embedding,
            'added_date': datetime.now().isoformat(),
            'additional_info': additional_info or {}
        }
        
        self.next_id += 1
        logger.info(f"Added face for {person_name} with ID {face_id}")
        return face_id
    
    def recognize_face(self, image: np.ndarray, threshold: float = 0.6) -> List[Dict]:
        """
        Recognize faces in an image
        
        Args:
            image: Input image
            threshold: Similarity threshold for recognition
            
        Returns:
            List of recognition results with person info and confidence
        """
        # Detect faces
        faces = self.detect_faces(image)
        
        if not faces or self.faiss_index is None:
            return []
        
        # Extract embeddings
        embeddings = self.extract_embeddings(image, faces)
        
        results = []
        for i, embedding in enumerate(embeddings):
            # Normalize embedding
            embedding_normalized = embedding / np.linalg.norm(embedding)
            
            # Search in FAISS index
            similarities, indices = self.faiss_index.search(
                embedding_normalized.reshape(1, -1), 
                min(10, self.faiss_index.ntotal)
            )  # type: ignore
            
            if len(indices[0]) > 0 and similarities[0][0] > threshold:
                best_match_id = indices[0][0]
                confidence = float(similarities[0][0])
                
                if best_match_id in self.face_database:
                    person_info = self.face_database[best_match_id]
                    results.append({
                        'face_index': i,
                        'person_id': best_match_id,
                        'name': person_info['name'],
                        'confidence': confidence,
                        'bbox': getattr(faces[i], 'bbox', None),
                        'landmarks': getattr(faces[i], 'kps', None)
                    })
                else:
                    results.append({
                        'face_index': i,
                        'person_id': None,
                        'name': 'Unknown',
                        'confidence': confidence,
                        'bbox': getattr(faces[i], 'bbox', None),
                        'landmarks': getattr(faces[i], 'kps', None)
                    })
            else:
                results.append({
                    'face_index': i,
                    'person_id': None,
                    'name': 'Unknown',
                    'confidence': 0.0,
                    'bbox': getattr(faces[i], 'bbox', None),
                    'landmarks': getattr(faces[i], 'kps', None)
                })
        
        return results
    
    def save_database(self, filepath: str):
        """Save the face database and FAISS index"""
        try:
            # Save FAISS index
            faiss.write_index(self.faiss_index, f"{filepath}_index.faiss")
            
            # Save metadata
            with open(f"{filepath}_metadata.pkl", 'wb') as f:
                pickle.dump({
                    'face_database': self.face_database,
                    'next_id': self.next_id
                }, f)
            
            logger.info(f"Database saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving database: {e}")
            raise
    
    def load_database(self, filepath: str):
        """Load the face database and FAISS index"""
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index(f"{filepath}_index.faiss")
            
            # Load metadata
            with open(f"{filepath}_metadata.pkl", 'rb') as f:
                data = pickle.load(f)
                self.face_database = data['face_database']
                self.next_id = data['next_id']
            
            logger.info(f"Database loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            raise
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the face database"""
        return {
            'total_faces': len(self.face_database),
            'index_size': self.faiss_index.ntotal if self.faiss_index else 0,
            'next_id': self.next_id,
            'persons': list(set([info['name'] for info in self.face_database.values()]))
        }
    
    def remove_face(self, face_id: int) -> bool:
        """
        Remove a face from the database
        
        Args:
            face_id: ID of the face to remove
            
        Returns:
            True if successful, False otherwise
        """
        if face_id not in self.face_database:
            return False
        
        # Remove from metadata
        del self.face_database[face_id]
        
        # Rebuild FAISS index
        self._rebuild_index()
        
        logger.info(f"Removed face with ID {face_id}")
        return True
    
    def _rebuild_index(self):
        """Rebuild FAISS index from current database"""
        if not self.face_database:
            self.faiss_index = None
            return
        
        # Create new index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add all embeddings
        embeddings = []
        for face_id, info in self.face_database.items():
            embedding = info['embedding']
            embedding_normalized = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding_normalized)
        
        if embeddings:
            embeddings_array = np.array(embeddings)
            self.faiss_index.add(embeddings_array)  # type: ignore
        
        logger.info("FAISS index rebuilt successfully")


def load_image(image_path: str) -> np.ndarray:
    """Load image from file path"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(image: np.ndarray, output_path: str):
    """Save image to file"""
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)


if __name__ == "__main__":
    # Example usage
    system = FaceRecognitionSystem()
    
    # Example: Add faces to database
    # face_id = system.add_face_to_database(image, "John Doe", {"age": 30, "department": "IT"})
    
    # Example: Recognize faces
    # results = system.recognize_face(image, threshold=0.6)
    
    # Example: Save database
    # system.save_database("face_database")
    
    print("Face Recognition System initialized successfully!")
    print("Use the methods to add faces, recognize faces, and manage the database.") 
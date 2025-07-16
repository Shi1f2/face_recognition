"""
Configuration file for the Face Recognition System
"""

# Model Configuration
MODEL_CONFIG = {
    'model_path': 'models',
    'embedding_dim': 512,
    'detection_size': (640, 640),
    'ctx_id': 0,  # 0 for GPU, -1 for CPU
    'model_name': 'buffalo_l'  # InsightFace model
}

# Recognition Configuration
RECOGNITION_CONFIG = {
    'default_threshold': 0.6,
    'max_results': 10,
    'min_face_size': 20
}

# Database Configuration
DATABASE_CONFIG = {
    'index_type': 'flat',  # 'flat' for IndexFlatIP, 'ivf' for IndexIVFFlat
    'normalize_embeddings': True,
    'save_embeddings': True
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'batch_size': 1,
    'enable_logging': True,
    'log_level': 'INFO'
}

# File Paths
PATHS = {
    'models_dir': 'models',
    'sample_images_dir': 'sample_images',
    'database_dir': '.',
    'logs_dir': 'logs'
}

# Image Processing
IMAGE_CONFIG = {
    'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp'],
    'max_image_size': (1920, 1080),
    'color_space': 'RGB'
} 
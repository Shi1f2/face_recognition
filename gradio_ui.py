import gradio as gr
import cv2
import numpy as np
import os
import tempfile
from datetime import datetime
from face_recognition_system import FaceRecognitionSystem, load_image, save_image

# Initialize the face recognition system
system = FaceRecognitionSystem()

def add_face_to_database(image, person_name, additional_info=""):
    """Add a face to the database"""
    try:
        if image is None:
            return "Please upload an image first.", None
        
        if not person_name.strip():
            return "Please enter a person name.", None
        
        # Convert PIL image to numpy array if needed
        if hasattr(image, 'convert'):
            image = np.array(image)
        
        # Add face to database
        face_id = system.add_face_to_database(
            image, 
            person_name.strip(),
            {"additional_info": additional_info} if additional_info else None
        )
        
        # Save database
        system.save_database("face_database")
        
        return f"✅ Successfully added {person_name} with ID {face_id} to database!", None
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def recognize_faces(image, threshold=0.6):
    """Recognize faces in an image"""
    try:
        if image is None:
            return "Please upload an image first.", None
        
        # Convert PIL image to numpy array if needed
        if hasattr(image, 'convert'):
            image = np.array(image)
        
        # Recognize faces
        results = system.recognize_face(image, threshold=threshold)
        
        if not results:
            return "No faces detected or no matches found.", None
        
        # Create result text
        result_text = f"Found {len(results)} face(s):\n\n"
        for i, result in enumerate(results):
            result_text += f"Face {i+1}:\n"
            result_text += f"  Name: {result['name']}\n"
            result_text += f"  Confidence: {result['confidence']:.3f}\n"
            if result.get('distance'):
                result_text += f"  Distance: {result['distance']:.3f}\n"
            result_text += "\n"
        
        return result_text, None
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def process_video(video_path, threshold=0.6, frame_interval=30, max_frames=100):
    """Process a video for face recognition and return HTML results"""
    try:
        if video_path is None:
            return "<div style='color:red;'>Please upload a video first.</div>", None, None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "<div style='color:red;'>❌ Error: Could not open video file.</div>", None, None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        frame_count = 0
        processed_frames = 0
        results_summary = {}
        detected_frames = []
        
        html = f"""
        <div style='font-family:monospace;'>
        <h3>📹 Video Info</h3>
        <ul>
            <li><b>Total frames:</b> {total_frames}</li>
            <li><b>FPS:</b> {fps:.2f}</li>
            <li><b>Duration:</b> {duration:.2f} seconds</li>
            <li><b>Processing every {frame_interval} frames</b></li>
        </ul>
        """
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0 and (max_frames == 0 or processed_frames < max_frames):
                frame_results = system.recognize_face(frame, threshold=threshold)
                if frame_results:
                    timestamp = frame_count / fps
                    detected_frames.append({
                        'frame': frame_count,
                        'timestamp': timestamp,
                        'results': frame_results
                    })
                    for result in frame_results:
                        name = result['name']
                        if name not in results_summary:
                            results_summary[name] = {
                                'count': 0,
                                'total_confidence': 0,
                                'first_seen': timestamp,
                                'last_seen': timestamp
                            }
                        results_summary[name]['count'] += 1
                        results_summary[name]['total_confidence'] += result['confidence']
                        results_summary[name]['last_seen'] = timestamp
                processed_frames += 1
            frame_count += 1
        cap.release()
        # Summary
        if detected_frames:
            html += f"<h3>✅ Found faces in {len(detected_frames)} frames</h3>"
            html += "<h4>📊 Summary by Person</h4><table border='1' cellpadding='4' style='border-collapse:collapse;'><tr><th>Name</th><th>Appearances</th><th>Avg Confidence</th><th>First Seen (s)</th><th>Last Seen (s)</th></tr>"
            for name, stats in results_summary.items():
                avg_confidence = stats['total_confidence'] / stats['count']
                html += f"<tr><td>{name}</td><td>{stats['count']}</td><td>{avg_confidence:.3f}</td><td>{stats['first_seen']:.2f}</td><td>{stats['last_seen']:.2f}</td></tr>"
            html += "</table>"
            # All frame-by-frame details
            html += "<h4>📋 Frame-by-frame details</h4>"
            html += "<div style='max-height:400px;overflow:auto;border:1px solid #ccc;padding:8px;'>"
            html += "<table border='1' cellpadding='4' style='border-collapse:collapse;'><tr><th>Frame</th><th>Timestamp (s)</th><th>Detections</th></tr>"
            for detection in detected_frames:
                dets = "<ul style='margin:0;padding-left:18px;'>"
                for result in detection['results']:
                    dets += f"<li><b>{result['name']}</b> (confidence: {result['confidence']:.3f})</li>"
                dets += "</ul>"
                html += f"<tr><td>{detection['frame']}</td><td>{detection['timestamp']:.2f}</td><td>{dets}</td></tr>"
            html += "</table></div>"
        else:
            html += "<div style='color:red;'>❌ No faces detected in the video.</div>"
        html += "</div>"
        # Sample frame
        sample_frame = None
        if detected_frames:
            first_detection = detected_frames[0]
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_detection['frame'])
            ret, sample_frame = cap.read()
            cap.release()
            if ret:
                for result in first_detection['results']:
                    cv2.putText(sample_frame, f"{result['name']} ({result['confidence']:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return html, sample_frame, None
    except Exception as e:
        return f"<div style='color:red;'>❌ Error processing video: {str(e)}</div>", None, None

def get_database_stats():
    """Get database statistics"""
    try:
        stats = system.get_database_stats()
        
        stats_text = f"📊 Database Statistics:\n\n"
        stats_text += f"Total faces: {stats['total_faces']}\n"
        stats_text += f"FAISS index size: {stats['index_size']}\n"
        stats_text += f"Next ID: {stats['next_id']}\n"
        stats_text += f"Persons in database: {', '.join(stats['persons']) if stats['persons'] else 'None'}\n"
        
        return stats_text
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def load_existing_database():
    """Load existing database"""
    try:
        if os.path.exists("face_database_index.faiss"):
            system.load_database("face_database")
            return "✅ Database loaded successfully!"
        else:
            return "ℹ️ No existing database found. Start by adding faces."
    except Exception as e:
        return f"❌ Error loading database: {str(e)}"

def clear_database():
    """Clear the database"""
    try:
        # Remove database files
        for ext in ['_index.faiss', '_metadata.pkl']:
            file_path = f"face_database{ext}"
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Reset system
        global system
        system = FaceRecognitionSystem()
        
        return "🗑️ Database cleared successfully!"
    except Exception as e:
        return f"❌ Error clearing database: {str(e)}"

def create_sample_image():
    """Create a sample image for testing"""
    # Create a simple colored image
    sample_img = np.zeros((300, 300, 3), dtype=np.uint8)
    sample_img[:] = (100, 150, 200)  # Blue-ish color
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(sample_img, 'SAMPLE', (80, 150), font, 1.5, (255, 255, 255), 3)
    
    return sample_img

# Create Gradio interface
with gr.Blocks(title="Face Recognition System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Face Recognition System")
    gr.Markdown("A comprehensive face recognition system using RetinaFace + InsightFace + FAISS")
    
    with gr.Tab("📥 Add Faces"):
        gr.Markdown("### Add a new face to the database")
        
        with gr.Row():
            with gr.Column():
                add_image = gr.Image(label="Upload Face Image", type="numpy")
                person_name = gr.Textbox(label="Person Name", placeholder="Enter the person's name")
                additional_info = gr.Textbox(label="Additional Info (Optional)", placeholder="Any additional information")
                add_button = gr.Button("Add Face to Database", variant="primary")
            
            with gr.Column():
                add_output = gr.Textbox(label="Result", lines=5)
                sample_button = gr.Button("Create Sample Image")
        
        add_button.click(
            fn=add_face_to_database,
            inputs=[add_image, person_name, additional_info],
            outputs=[add_output, add_image]
        )
        
        sample_button.click(
            fn=create_sample_image,
            outputs=add_image
        )
    
    with gr.Tab("🔍 Recognize Faces"):
        gr.Markdown("### Recognize faces in an image")
        
        with gr.Row():
            with gr.Column():
                recognize_image = gr.Image(label="Upload Image to Recognize", type="numpy")
                threshold = gr.Slider(
                    minimum=0.1, 
                    maximum=1.0, 
                    value=0.6, 
                    step=0.1, 
                    label="Recognition Threshold (higher = more strict)"
                )
                recognize_button = gr.Button("Recognize Faces", variant="primary")
            
            with gr.Column():
                recognize_output = gr.Textbox(label="Recognition Results", lines=10)
                sample_recog_button = gr.Button("Create Sample Image")
        
        recognize_button.click(
            fn=recognize_faces,
            inputs=[recognize_image, threshold],
            outputs=[recognize_output, recognize_image]
        )
        
        sample_recog_button.click(
            fn=create_sample_image,
            outputs=recognize_image
        )
    
    with gr.Tab("🎥 Process Video"):
        gr.Markdown("### Process a video for face recognition")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video for Processing")
                threshold = gr.Slider(
                    minimum=0.1, 
                    maximum=1.0, 
                    value=0.6, 
                    step=0.1, 
                    label="Recognition Threshold (higher = more strict)"
                )
                frame_interval = gr.Slider(
                    minimum=1, 
                    maximum=100, 
                    value=30, 
                    step=1, 
                    label="Frame Interval (process every N frames)"
                )
                # The max_frames slider will be updated dynamically based on the uploaded video
                max_frames = gr.Slider(
                    minimum=0, 
                    value=0, 
                    step=1, 
                    label="Max Frames to Process (0 for all)"
                )
                process_button = gr.Button("Process Video", variant="primary")
            
            with gr.Column():
                video_output_html = gr.HTML(label="Video Processing Results")
                video_output_frame = gr.Image(label="Sample Frame with Detections", visible=False)
        
        def update_max_frames_slider(video_path):
            if video_path is None:
                return gr.Slider(value=0, minimum=0, step=1, label="Max Frames to Process (0 for all)")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return gr.Slider(value=0, minimum=0, step=1, label="Max Frames to Process (0 for all)")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return gr.Slider(maximum=total_frames, value=total_frames, minimum=0, step=1, label="Max Frames to Process (0 for all)")

        video_input.change(
            fn=update_max_frames_slider,
            inputs=video_input,
            outputs=max_frames
        )

        process_button.click(
            fn=process_video,
            inputs=[video_input, threshold, frame_interval, max_frames],
            outputs=[video_output_html, video_output_frame]
        )
    
    with gr.Tab("📊 Database Management"):
        gr.Markdown("### Manage the face database")
        
        with gr.Row():
            with gr.Column():
                load_db_button = gr.Button("Load Database", variant="secondary")
                stats_button = gr.Button("Show Statistics", variant="secondary")
                clear_db_button = gr.Button("Clear Database", variant="stop")
            
            with gr.Column():
                db_output = gr.Textbox(label="Database Info", lines=8)
        
        load_db_button.click(
            fn=load_existing_database,
            outputs=db_output
        )
        
        stats_button.click(
            fn=get_database_stats,
            outputs=db_output
        )
        
        clear_db_button.click(
            fn=clear_database,
            outputs=db_output
        )
    
    with gr.Tab("ℹ️ Help"):
        gr.Markdown("""
        ## How to Use This System
        
        ### Adding Faces
        1. Upload a clear image of a person's face
        2. Enter their name
        3. Optionally add additional information
        4. Click "Add Face to Database"
        
        ### Recognizing Faces
        1. Upload an image containing faces
        2. Adjust the recognition threshold if needed
        3. Click "Recognize Faces"
        
        ### Processing Videos
        1. Upload a video file (MP4, AVI, MOV, etc.)
        2. Adjust the recognition threshold if needed
        3. Set frame interval (process every N frames for speed)
        4. Set maximum frames to process (0 for all frames)
        5. Click "Process Video"
        
        ### Database Management
        - **Load Database**: Load previously saved faces
        - **Show Statistics**: View database information
        - **Clear Database**: Remove all stored faces
        
        ### Tips
        - Use clear, well-lit images for better recognition
        - Higher threshold values require more similarity for a match
        - The system automatically saves the database after adding faces
        - Multiple faces can be recognized in a single image
        - For videos, use higher frame intervals for faster processing
        - Video processing shows summary statistics and frame-by-frame details
        
        ### Supported Video Formats
        - MP4, AVI, MOV, MKV, WMV, FLV, and other common formats
        - Processing speed depends on video length and frame interval
        
        ### Technical Details
        - Face Detection: RetinaFace
        - Face Recognition: InsightFace (ArcFace + iResNet100)
        - Storage: FAISS for efficient similarity search
        - Embedding Dimension: 512
        - Video Processing: OpenCV for frame extraction
        """)
    
    # Auto-load database on startup
    gr.Markdown("---")
    gr.Markdown("*System initialized. Database will be loaded automatically if available.*")

if __name__ == "__main__":
    # Try to load existing database
    try:
        if os.path.exists("face_database_index.faiss"):
            system.load_database("face_database")
            print("✅ Database loaded successfully!")
    except Exception as e:
        print(f"ℹ️ No existing database found or error loading: {e}")
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    ) 
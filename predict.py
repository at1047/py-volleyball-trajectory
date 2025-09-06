from ultralytics import YOLO
import os
import cv2
import numpy as np
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load the trained model
# model_path = 'yolo12n.pt'
model_path = 'runs/detect/ball_detection3/weights/best.pt'  # or 'last.pt' for the last checkpoint
model = YOLO(model_path)

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

def process_video(video_path, output_path=None):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create frames directory
    frames_dir = os.path.join(os.path.dirname(output_path), 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    frame_count = 0
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:  # Print progress every 10 frames
            print(f"Processing frame {frame_count}/{total_frames}")

        # Run prediction
        results = model.predict(
            source=frame,
            conf=0.1,  # confidence threshold for detection
            save=False,  # we'll save manually
        )
        
        # Process detections
        detections = []
        detection_confs = {}  # Store confidence scores for each detection
        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())  # Get class ID
                    cls_name = model.names[cls]  # Get class name
                    
                    # Draw low confidence detections without tracking
                    cv2.rectangle(frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 0, 255), 2)  # Red for low confidence
                    
                    label = f"{cls_name} Conf: {conf:.2f}"
                    cv2.putText(frame, 
                              label, 
                              (int(x1), int(y1-10)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.9, 
                              (0, 0, 255), 
                              2)
        
        # Update tracker only for high confidence detections
        # if detections:
        #     tracks = tracker.update_tracks(detections, frame=frame)
            
        #     # Draw tracking results
        #     for track in tracks:
        #         if not track.is_confirmed():
        #             continue
                
        #         track_id = track.track_id
        #         ltrb = track.to_ltrb()
                
        #         # Find the closest detection to get its confidence
        #         best_conf = 0.0
        #         for det, conf in detection_confs.items():
        #             det_ltrb = [det[0], det[1], det[2], det[3]]
        #             # Calculate IoU between track and detection
        #             iou = calculate_iou(ltrb, det_ltrb)
        #             if iou > 0.5 and conf > best_conf:  # If significant overlap
        #                 best_conf = conf
                
        #         # Draw bounding box
        #         cv2.rectangle(frame, 
        #                     (int(ltrb[0]), int(ltrb[1])), 
        #                     (int(ltrb[2]), int(ltrb[3])), 
        #                     (0, 255, 0), 2)  # Green for tracked objects
                
        #         # Draw track ID and confidence
        #         label = f"ID: {track_id} Conf: {best_conf:.2f}"
        #         cv2.putText(frame, 
        #                   label, 
        #                   (int(ltrb[0]), int(ltrb[1]-10)), 
        #                   cv2.FONT_HERSHEY_SIMPLEX, 
        #                   0.9, 
        #                   (0, 255, 0), 
        #                   2)

        # Save individual frame
        frame_path = os.path.join(frames_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_path, frame)

        # Write frame to output video
        if output_path:
            out.write(frame)

        # Display frame (optional)
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    # Release resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

    print(f"Video processing complete. Processed {frame_count} frames.")
    print(f"Frames saved to: {frames_dir}")

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes."""
    # Get coordinates of intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    iou = intersection / float(box1_area + box2_area - intersection)
    return iou

if __name__ == "__main__":
    # Specify your video path here
    video_path = "datasets/My Videos/IMG_4334 2.mov"
    
    # Create output directory
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output video path
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f'tracked_{video_name}')
    
    # Process the video
    process_video(video_path, output_path) 
import os
import cv2
import numpy as np
from super_gradients.training import models
from super_gradients.common.object_names import  Models
from deepsort import deepSORT_Tracker 
model = models.get("yolo_nas_m", pretrained_weights="coco")
import warnings
warnings.filterwarnings("ignore")

video_path = os.path.join('.', 'data', 'UIT_01.mp4')
video_out_path = os.path.join('.', 'ds-UIT-01.mp4')
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap_out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height)) 
tracker = deepSORT_Tracker()
detection_threshold = 0.3
output_folder = os.path.splitext(video_out_path)[0]  
os.makedirs(output_folder, exist_ok=True) 
frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame)
    detections = []
    for result in results:
        for i, r in enumerate(result.prediction.bboxes_xyxy):
            x1, y1, x2, y2 = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            score = result.prediction.confidence[i]
            labels = result.prediction.labels[i]
            class_id = int(labels)
            if class_id == 0 and score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])
    tracker.update(frame, detections)
    
    track_ids = []
    for track in tracker.tracks:
        track_id, bbox = track
        x1, y1, x2, y2 = bbox
        track_ids.append(track_id)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (204, 0, 102), 3)
        cv2.putText(frame, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    for track in tracker.tracker.tracks:
        if track.track_id not in track_ids:
            tracker.tracker.tracks.remove(track)        
        
    cap_out.write(frame)
    print(f'Finished frame {frame_id}')
    output_path = os.path.join(output_folder, f'frame_{frame_id}.jpg')
    cv2.imwrite(output_path, frame)
    frame_id += 1

cap.release()
cap_out.release()
cv2.destroyAllWindows()

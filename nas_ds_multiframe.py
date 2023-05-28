import os
import cv2
import numpy as np
import torch
from super_gradients.training import models
from super_gradients.common.object_names import  Models
from deepsort import deepSORT_Tracker 
import warnings
warnings.filterwarnings("ignore")

model = models.get("yolo_nas_m", pretrained_weights="coco")
frame_dir_path = r"datasets/MOT17/train/MOT17-13-SDP/img1"
video_out_path = os.path.join('.', 'nas-ds-MOT17-13.mp4')
frame_paths = sorted([os.path.join(frame_dir_path, f) for f in os.listdir(frame_dir_path) if f.endswith('.jpg')])
first_frame = cv2.imread(frame_paths[0])
height, width, _ = first_frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 24
cap_out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
tracker = deepSORT_Tracker()
detection_threshold = 0.3
frame_id = 0
output_folder = os.path.splitext(video_out_path)[0]  
os.makedirs(output_folder, exist_ok=True) 

for frame_path in frame_paths:
    frame = cv2.imread(frame_path)
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
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (204, 0, 102), 3)
    tracker.update(frame, detections)
        
    for track in tracker.tracks:
        track_id, bbox = track
        x1, y1, x2, y2 = bbox
        cv2.putText(frame, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
    cap_out.write(frame)
    print(f'Finished frame {frame_id}')
    output_path = os.path.join(output_folder, f'frame_{frame_id}.jpg')
    cv2.imwrite(output_path, frame)
    frame_id += 1
cap_out.release()
cv2.destroyAllWindows()

import os
import cv2
import numpy as np
import torch
from strongsort.strong_sort import StrongSORT
import pathlib
from super_gradients.training import models
from super_gradients.common.object_names import  Models
import warnings
warnings.filterwarnings("ignore")


frame_dir_path = r"datasets/MOT17/test/MOT17-12-SDP/img1"
video_out_path = os.path.join('.', 'nas-ss-MOT17-12.mp4')
frame_paths = sorted([os.path.join(frame_dir_path, f) for f in os.listdir(frame_dir_path) if f.endswith('.jpg')])
first_frame = cv2.imread(frame_paths[0])
height, width, _ = first_frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
cap_out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
tracker = StrongSORT(model_weights=pathlib.Path('osnet_ain_x1_0_imagenet.pt'), device='cpu')
output_folder = os.path.splitext(video_out_path)[0]  
os.makedirs(output_folder, exist_ok=True) 
model = models.get("yolo_nas_m", pretrained_weights="coco")
detection_threshold = 0.4 
frame_id = 0

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
                detections.append([x1, y1, x2, y2, score, class_id])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (204, 0, 102), 3)
    tracker.update(torch.Tensor(detections), frame)
        
    for track in tracker.tracker.tracks:
        track_id = track.track_id
        box = track.to_tlwh()
        x1, y1, x2, y2 = tracker._tlwh_to_xyxy(box)
        cv2.putText(frame, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
    cap_out.write(frame)
    print(f'Finished frame {frame_id}')
    output_path = os.path.join(output_folder, f'frame_{frame_id}.jpg')
    cv2.imwrite(output_path, frame)
    frame_id += 1

cap_out.release()
cv2.destroyAllWindows()

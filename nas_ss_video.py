import os
import cv2
import numpy as np
import torch
from strongsort.strong_sort import StrongSORT
import pathlib
from super_gradients.training import models
from super_gradients.common.object_names import  Models
import warnings

model = models.get("yolo_nas_m", pretrained_weights="coco")
warnings.filterwarnings("ignore")
video_path = os.path.join('.', 'data', 'people.mp4')
video_out_path = os.path.join('.', 'nas-ss-people.mp4')
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap_out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
tracker = StrongSORT(model_weights=pathlib.Path('osnet_ain_x1_0_imagenet.pt'), device='cpu')
detection_threshold = 0.3
frame_id = 0
output_folder = os.path.splitext(video_out_path)[0]  
os.makedirs(output_folder, exist_ok=True) 

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
                detections.append([x1, y1, x2, y2, score, class_id])
            if class_id == 0:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (102, 0, 204), 2)
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

cap.release()
cap_out.release()
cv2.destroyAllWindows()

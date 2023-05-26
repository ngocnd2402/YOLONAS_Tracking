import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from strongsort.strong_sort import StrongSORT
import pathlib

model = YOLO("yolov5mu.pt",  task='detect')
video_path = os.path.join('../..', 'data', 'people.mp4')
video_out_path = os.path.join('.', 'v5m_ss_people_out.mp4')
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap_out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
tracker = StrongSORT(model_weights=pathlib.Path('osnet_x0_25_msmt17.pt'), device='cpu')
detection_threshold = 0.3
pts = {}
frame_id = 0
output_folder = os.path.splitext(video_out_path)[0]  
os.makedirs(output_folder, exist_ok=True) 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, verbose = False)
    for result in results:
        detections = []
        for r in result.boxes.data.tolist(): 
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if class_id == 0 and score > detection_threshold:
                detections.append([x1, y1, x2, y2, score, class_id])
        tracker.update(torch.Tensor(detections), frame)
        for track in tracker.tracker.tracks:
            track_id = track.track_id
            box = track.to_tlwh()
            x1, y1, x2, y2 = tracker._tlwh_to_xyxy(box)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (128, 0, 128), 2)
            cv2.putText(frame, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            
            if track_id not in pts:
                pts[track_id] = [(int((x1 + x2) / 2), int(y2))]
            else:
                pts[track_id].append((int((x1 + x2) / 2), int(y2)))

            color = (102, 0, 204)
            for j in range(1, len(pts[track_id])):
                if pts[track_id][j - 1] is None or pts[track_id][j] is None:
                    continue
                thickness = max(1, int(np.sqrt(64 / float(j + 1)) * 2))
                cv2.line(frame, pts[track_id][j - 1], pts[track_id][j], color, thickness)
    cap_out.write(frame)
    print(f'Processed frame {frame_id}')
    output_path = os.path.join(output_folder, f'frame_{frame_id}.jpg')
    cv2.imwrite(output_path, frame)
    frame_id += 1
    
cap.release()
cap_out.release()
cv2.destroyAllWindows()
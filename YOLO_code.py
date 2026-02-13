from ultralytics import YOLO
import numpy as np
import openvino


model = YOLO("my_model.pt")

model.export(format=openvino)

def run_YOLO(frame):

    #filters out any data that is less than 50% confident according to the model
    results = model.predict(frame, conf = 0.5)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "class": model.names[cls],
                "confidence": conf
            })

    return detections
    
def get_object_depth(depth_map, bbox):
    #create the bounds for hte box based on teh dimensions of the object
    x1, y1, x2, y2 = bbox

    roi = depth_map[y1:y2, x1:x2]
    roi = roi[np.isfinite(roi)]

    if roi.size == 0:
        return None
    #this returns a stable depth to us so that we can see it
    return np.median(roi)

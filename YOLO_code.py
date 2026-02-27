from ultralytics import YOLO
import numpy as np

model = YOLO("my_model.pt")

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

    pad = 0.2
    dx = int((x2 - x1))
    dy = int((y2 - y1))

    roi = depth_map[y1+dy:y2-dy, x1+dx:x2-dx]

    roi = roi[np.isfinite(roi)]
    z = roi[:,:,2]

    X = roi[:,0]
    Y = roi[:,1]
    Z = roi[:,2]

    width = np.percentile(X, 95) - np.percentile(X, 5)
    height = np.percentile(Y, 95) - np.percentile(Y,5)
    dep = np.percentile(Z, 95) - np.percentile(Z, 5)

    if roi.size < 50:
        return None

    z = np.median(roi)
    mad = np.median(np.abs(roi - z))

    roi = roi[np.abs(roi - z) < 3 * mad]

    depth = np.mean(roi)

    print(f"""
    Cube:
      width  = {width:.3f} m
      height = {height:.3f} m
      depth  = {dep:.3f} m
    """)

    return depth, width, height, dep

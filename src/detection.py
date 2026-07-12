import sys
import importlib
try:
    import cv2  # type: ignore[import]
except ImportError:
    sys.exit("Error: OpenCV (cv2) is not installed. Install it with 'pip install opencv-python'.")

try:
    ultralytics = importlib.import_module("ultralytics")
    YOLO = ultralytics.YOLO
except ImportError:
    sys.exit("Error: ultralytics is not installed. Install it with 'pip install ultralytics'.")

# Vehicle detection model
vehicle_model = YOLO("yolov8n.pt")

# License plate detection model
plate_model = YOLO("best.pt")
plate_results = None
vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle", "train"]

# Input video
cap = cv2.VideoCapture("traffic.mp4")

cv2.namedWindow("Vehicle + Plate Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vehicle + Plate Detection", 1280, 720)

if not cap.isOpened():
    sys.exit("Error: Could not open video file 'traffic.mp4'.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    print(frame.shape)

    # Vehicle Detection
    vehicle_results = vehicle_model(frame)
    for result in vehicle_results:
        for box in result.boxes:
            # box.cls may be a tensor/array; handle accordingly
            try:
                cls = int(box.cls[0])
            except Exception:
                cls = int(box.cls)
            class_name = vehicle_model.names.get(cls, str(cls))
            if class_name in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                plate = frame[y1:y2, x1:x2]

    # License Plate Detection
    plate_results = plate_model(frame, imgsz=1280, conf=0.1)
    for result in plate_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Vehicle + Plate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

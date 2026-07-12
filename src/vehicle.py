import cv2  # type: ignore[import]
from ultralytics import YOLO  # type: ignore[import]

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Vehicle classes to detect
vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle", "train"]

# Open video
cap = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = cap.read()


    if not ret:
        break
    
    # Run YOLO prediction
    results = model(frame)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]

            if class_name in vehicle_classes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0])

                label = f"{class_name} {confidence:.2f}"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
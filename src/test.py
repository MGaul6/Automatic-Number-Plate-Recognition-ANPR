from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        conf=0.10,
        imgsz=1280
    )

    print("Detected:", len(results[0].boxes))

    annotated = results[0].plot()

    cv2.imshow("YOLO Plate Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
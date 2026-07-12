import os

print("=" * 50)
print("RUNNING ANPR_CORRECTED")
print("File:", os.path.abspath(__file__))
print("=" * 50)

import cv2  # type: ignore[import]
import re
try:
    import easyocr  # type: ignore[import]
except ImportError as e:
    raise SystemExit("Missing dependency: easyocr. Install with `pip install easyocr`") from e
try:
    from ultralytics import YOLO  # type: ignore[import]
except ImportError as e:
    raise SystemExit("Missing dependency: ultralytics. Install with `pip install ultralytics`") from e

vehicle_model=YOLO("yolov8n.pt")
plate_model=YOLO("best.pt")
reader=easyocr.Reader(["en"])

vehicle_classes=["car","truck","bus","motorcycle","bicycle","train"]
# use 0 for webcam
#cap = cv2.VideoCapture("vehicles.mp4")
video_path = "vehicles.MOV"
cap = cv2.VideoCapture(video_path)

print("Video opened:", cap.isOpened())

if not cap.isOpened():
    raise SystemExit("Cannot open video.")

print("Camera opened:", cap.isOpened())
if not cap.isOpened():
    raise SystemExit("Cannot open input")

w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=cap.get(cv2.CAP_PROP_FPS)
if fps==0:
    fps=30

out = cv2.VideoWriter("webcam_output.avi",
                      cv2.VideoWriter_fourcc(*"XVID"),
                      fps,
                      (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        break
    cv2.putText(
    frame,
    "LIVE WEBCAM",
    (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.5,
    (0, 0, 255),
    3
)
    if not ret:
        break

    vehicle_results=vehicle_model(frame)
    for result in vehicle_results:
        for box in result.boxes:
            cls=int(box.cls[0])
            name=vehicle_model.names[cls]
            if name not in vehicle_classes:
                continue
            x1,y1,x2,y2=map(int,box.xyxy[0])
            conf=float(box.conf[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{name} {conf:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    plate_results=plate_model(frame,conf=0.10,imgsz=1280)
    print("Detected Plates:",len(plate_results[0].boxes))

    for result in plate_results:
        for box in result.boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            plate=frame[y1:y2,x1:x2]
            if plate.size==0:
                continue
            gray=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
            gray=cv2.resize(gray,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
            _,gray=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            text=""
            ocr=reader.readtext(gray)
            if ocr and ocr[0][2]>0.5:
                text=re.sub(r"[^A-Za-z0-9]","",ocr[0][1])
            cv2.putText(frame,text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.imshow("ANPR_CORRECTED",frame)
    #ut.write(frame)
    if cv2.waitKey(1)&0xFF==ord("q"):
        break

cap.release()
#ut.release()
cv2.destroyAllWindows()

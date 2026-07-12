import cv2
import easyocr
import re
from ultralytics import YOLO

# Load trained license plate model
model = YOLO("best.pt")

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Input video
video_path = "traffic.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
    "output_video.avi",
    fourcc,
    fps,
    (width, height)
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run license plate detection
    results = model(
        frame,
        conf=0.10,
        imgsz=1280
    )

    frame_height, frame_width = frame.shape[:2]

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_width - 1, x2)
            y2 = min(frame_height - 1, y2)

            # Crop plate
            plate = frame[y1:y2, x1:x2]

            plate_text = ""

            if plate.size != 0:

                # OCR
                ocr_result = reader.readtext(plate)

                if len(ocr_result) > 0:

                    confidence = ocr_result[0][2]

                    if confidence > 0.5:

                        plate_text = re.sub(
                            r'[^A-Za-z0-9]',
                            '',
                            ocr_result[0][1]
                        )

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Show recognized text
            cv2.putText(
                frame,
                plate_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

    # Display
    cv2.imshow("ANPR - License Plate Recognition", frame)

    # Save output
    out.write(frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Detection completed.")
print("Output saved as output_video.avi")
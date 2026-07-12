import cv2

cap = cv2.VideoCapture("traffic.mp4")   # replace with your video name

ret, frame = cap.read()

if ret:
    cv2.imwrite("test_frame.jpg", frame)
    print("Frame saved successfully")
else:
    print("Could not read video")

cap.release()
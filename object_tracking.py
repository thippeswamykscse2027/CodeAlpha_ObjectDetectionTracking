import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detection + Tracking
    results = model.track(
        frame,
        persist=True
    )

    annotated_frame = results[0].plot()

    cv2.imshow(
        "Object Detection and Tracking",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
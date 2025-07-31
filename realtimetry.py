import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO('C:/Users/Rutuja Navale/Desktop/PBLFINAL/my_model/my_model.pt')

# Initialize both cameras
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Failed to grab frames from both cameras")
        break

    # Run inference on both frames
    results1 = model(frame1, conf=0.5)
    results2 = model(frame2, conf=0.5)

    # Draw bounding boxes
    frame1_boxed = results1[0].plot()
    frame2_boxed = results2[0].plot()

    # Check for 'Damaged' label in both detections
    is_defective = False

    for box in results1[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower()
        print(f"Camera 1 detected: {label}")
        if label == "damaged":  # Make sure this matches your actual label
            is_defective = True
            break

    if not is_defective:
        for box in results2[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()
            print(f"Camera 2 detected: {label}")
            if label == "damaged":  # Make sure this matches your actual label
                is_defective = True
                break

    # Combine both frames side by side
    combined = np.hstack((frame1_boxed, frame2_boxed))

    # Label decision on combined image
    final_label = "DEFECTIVE PACKAGE" if is_defective else "GOOD PACKAGE"
    color = (0, 0, 255) if is_defective else (0, 255, 0)
    cv2.putText(combined, final_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Show combined result
    cv2.imshow("Package Inspection - Dual View", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap1.release()
cap2.release()
cv2.destroyAllWindows()

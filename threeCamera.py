import cv2
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO('C:/Users/Rutuja Navale/Desktop/PBLFINAL/my_model/my_model.pt')

# Class ID for 'defect' (update according to your model)
defect_class_id = 1

# Open both camera sources
camera_1 = cv2.VideoCapture(0)  # Laptop camera
camera_2 = cv2.VideoCapture(1)  # Logitech camera

# Function to classify package based on multiple cameras
def classify_package_and_display(cams, display_cam_index=0):
    classifications = []
    display_frame = None

    for i, cam in enumerate(cams):
        ret, frame = cam.read()
        if not ret:
            print(f"Camera {i+1} failed to grab frame")
            classifications.append(False)
            continue

        # Perform detection
        results = model(frame, conf=0.5)
        boxes = results[0].boxes

        # Check for defect class
        if boxes.cls is not None:
            detected_classes = boxes.cls.tolist()
            is_defective = defect_class_id in detected_classes
        else:
            is_defective = False

        classifications.append(is_defective)

        # Save the display frame from the selected camera only
        if i == display_cam_index:
            display_frame = results[0].plot()

    # Determine final result
    final_result = "Defect" if any(classifications) else "Good"

    return final_result, display_frame

# Main loop
while True:
    # Run detection + get classification and the single frame to display
    result, frame_to_display = classify_package_and_display([camera_1, camera_2], display_cam_index=0)

    # Show only one camera frame with result overlay
    if frame_to_display is not None:
        cv2.putText(frame_to_display, f"Classification: {result}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if result == "Good" else (0, 0, 255), 2)
        cv2.imshow("Package Defect Detection", frame_to_display)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
camera_1.release()
camera_2.release()
cv2.destroyAllWindows()

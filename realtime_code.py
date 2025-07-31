import cv2
from ultralytics import YOLO

# Load the YOLO model using the direct model loading method
model = YOLO('C:/Users/Rutuja Navale/Desktop/PBLFINAL/my_model/my_model.pt')  

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference on the frame
    results = model(frame, conf=0.5)

    # Check if there are any detections in the results
    if len(results) > 0:
        # Render results (draw the bounding boxes on the frame)
        frame_with_boxes = results[0].plot()  # Use the first result in the list
    else:
        frame_with_boxes = frame  # No detections, show the original frame
   

    # Display the resulting frame
    cv2.imshow("YOLO Real-Time Detection", frame_with_boxes)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

import os
from ultralytics import YOLO
import cv2

# Load the image
image_path = "D:/YOLO Object detection/test/9ff76000-f8ed-11ee-80f2-18473dba8b8a.jpg"  # Change this to your image path
frame = cv2.imread(image_path)

# Ensure the image is loaded correctly
if frame is None:
    print("Error: Could not load image.")
else:
    # Get image dimensions
    H, W = frame.shape[:2]  # Use [:2] to get height and width

    # Load the model
    model_path = "D:/YOLO Object detection/weights/last.pt"
    model = YOLO(model_path)  # Load a custom model

    # Define the detection threshold
    threshold = 0.5

    # Perform inference on the image
    results = model(frame)[0]

    # Draw bounding boxes and labels on the image
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Save the output image
    output_path = "D:/YOLO Object detection/test/detected_image3.jpg"  # Change this to your desired output path
    cv2.imwrite(output_path, frame)

    # Optionally, display the image with detections
    cv2.imshow('Detected Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

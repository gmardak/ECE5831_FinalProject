import cv2
import time
import tempfile
import playsound  # Make sure this library is installed using 'pip install playsound'
from roboflow import Roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="-") # omitted api key
project = rf.workspace("university-of-michigan-w3ucb").project("footpath-detection-h3aok")
model = project.version(2).model

cap = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0
fps = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break

    new_frame_time = time.time()

    # Display frame every 1 second (1 FPS)
    if (new_frame_time - prev_frame_time) > 1/fps:
        # Save the frame to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image_file:
            cv2.imwrite(temp_image_file.name, frame)
        
            # Send the frame to Roboflow for prediction and save the result
            prediction = model.predict(temp_image_file.name, confidence=10, overlap=30)
            prediction_path = temp_image_file.name.replace(".jpg", "_prediction.jpg")
            prediction.save(prediction_path)

            # Check if any bounding boxes are detected
            if len(prediction.json()["predictions"]) > 0:
                for pred in prediction.json()["predictions"]:
                    x, y, width, height = pred["x"], pred["y"], pred["width"], pred["height"]
                    center_x, center_y = x + width / 2, y + height / 2

                    # Compare with the center of the image
                    img_center_x = frame.shape[1] / 2
                    if center_x < img_center_x:
                        # Play sound for left
                        playsound.playsound('left.mp3')
                    else:
                        # Play sound for right
                        playsound.playsound('right.mp3')

        # Read the predicted image for displayq
        img_np = cv2.imread(prediction_path)

        cv2.imshow('Object Detection', img_np)
        prev_frame_time = new_frame_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

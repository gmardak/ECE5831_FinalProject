FinalCode.py

import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import pygame
import pygame.mixer
import supervision as sv

class ObjectDetection:
    def __init__(self, capture_index):
        # Initialize pygame for audio handling
        pygame.init()

        # Set video capture index
        self.capture_index = capture_index

        # Choose the computing device (CUDA if available, otherwise CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        # Load the model
        self.model = self.load_model()

        # Get class names from the model
        self.CLASS_NAMES_DICT = self.model.model.names

        # Initialize the box annotator for drawing bounding boxes
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)

        # Flag to check if audio is currently playing
        self.audio_playing = False

    def load_model(self):
        # Load a pretrained YOLO model
        model = YOLO("best.pt")
        model.fuse()  # Optimize the model
        return model

    def predict(self, frame):
        # Perform object detection on the frame
        return self.model(frame)

    def plot_bboxes(self, results, frame):
        # Calculate the center of the camera frame
        camera_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        # Process each detection result
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.cpu().numpy()

                # Process each detected box
                for box in boxes:
                    class_id = int(box.cls[0])
                    # Check if detected class is 'footpath'
                    if class_id == 0:
                        xyxy = box.xyxy[0]
                        # Calculate the centroid of the bounding box
                        centroid = ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)

                        # Play different audio based on the position of the centroid
                        if centroid[0] + 50 < camera_center[0]:
                            self.play_audio('audio_left.mp3')  # Centroid is to the left
                        elif centroid[0] - 50 > camera_center[0]:
                            self.play_audio('audio_right.mp3')  # Centroid is to the right
                        else:
                            self.play_audio('audio_center.mp3')  # Centroid is approximately at the center

                        # Annotate the frame with bounding box and label
                        label = f"{self.CLASS_NAMES_DICT[class_id]} {box.conf[0]:.2f}"
                        xyxy_2d = np.expand_dims(xyxy, axis=0)
                        class_id_1d = np.array([class_id])
                        conf_1d = np.array([box.conf[0]])
                        frame = self.box_annotator.annotate(scene=frame, 
                                                            detections=sv.Detections(xyxy=xyxy_2d, 
                                                                                    confidence=conf_1d, 
                                                                                    class_id=class_id_1d), 
                                                            labels=[label])

        return frame

    def play_audio(self, audio_file):
        # Play audio if not already playing
        if not self.audio_playing:
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            self.audio_playing = True
            pygame.mixer.music.set_endevent(pygame.USEREVENT)
            pygame.event.set_allowed(pygame.USEREVENT)

    def check_audio_finished(self):
        # Check if the audio has finished playing
        for event in pygame.event.get(pygame.USEREVENT):
            if event.type == pygame.USEREVENT:
                self.audio_playing = False  # Reset the flag

    def __call__(self):
        # Initialize video capture
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Main loop for video processing
        while True:
            self.check_audio_finished()  # Check if the audio has finished

            start_time = time()
            
            # Read a frame from the video capture
            ret, frame = cap.read()
            assert ret
            
            # Predict and annotate the frame
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)
            
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
             
            # Display FPS on the frame
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow('YOLOv8 Detection', frame)
 
            # Break the loop if 'ESC' is pressed
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        

detector = ObjectDetection(capture_index=0)
detector()

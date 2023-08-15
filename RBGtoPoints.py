import os
import cv2
import mediapipe as mp

# Define paths
dataset_dir = "D:/College/6th Sem/CIP/Data set/GaitDatasetB-silh/001/001/bg-01/000"  # Change this to the path of the CASIA-B dataset
output_dir = "D:/College/6th Sem/CIP/Data set/RGBdataset/001/001/bg-01/000"  # Change this to the path of the directory where you want to save the results

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Loop through all the images in the dataset
for root, dirs, files in os.walk(dataset_dir):
    for filename in files:
        # Load image
        image_path = os.path.join(root, filename)
        image = cv2.imread(image_path)

        # Resize image
        image = cv2.resize(image, (4000,3000))

        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe pose model
        results = pose_model.process(image_rgb)

        # Extract pose landmarks
        pose_landmarks = results.pose_landmarks

        # Draw pose landmarks on image
        mp_drawing = mp.solutions.drawing_utils
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Save annotated image
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, annotated_image)

# Clean up
pose_model.close()

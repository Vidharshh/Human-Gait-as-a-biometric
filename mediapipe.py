import cv2
import mediapipe as mp

# Load the MediaPipe Pose model
mp_pose = mp.solutions.pose

# Load the sample image
image = cv2.imread('.jpg')

# Convert the image to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize the pose detection module
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detection:
    # Run pose detection on the image
    results = pose_detection.process(image_rgb)

    # Extract the landmarks from the pose detection results
    landmarks = results.pose_landmarks

    # If landmarks were detected, draw them on the image and save it
    if landmarks is not None:
        # Draw the landmarks on the image
        mp_drawing = mp.solutions.drawing_utils
        image_annotated = image.copy()
        mp_drawing.draw_landmarks(image_annotated, landmarks, mp_pose.POSE_CONNECTIONS)

        # Save the annotated image
        cv2.imwrite('annotated_image.jpg', image_annotated)

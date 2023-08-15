import os

import cv2
import mediapipe as mp
# Load video file
cap = cv2.VideoCapture("C:\\Users\\Dwaa2\\Dropbox\\My PC (Dwaarakesh)\\Downloads\\walk1.mp4")

# Create Mediapipe hands object
mp_hands = mp.solutions.hands.Hands()

# Create output directory if it doesn't exist
output_dir = 'frames_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through video frames
while True:
    # Read next frame
    ret, frame = cap.read()
    ret, frame = cap.read()
    print(ret)
    ret, frame = cap.read()
    print(frame.shape)

    # Convert frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Mediapipe hands object
    results = mp_hands.process(frame)

    # Check if hands were detected
    if results.multi_hand_landmarks:
        # Loop through detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            # Loop through landmarks
            for landmark in hand_landmarks.landmark:
                # Extract landmark coordinates
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                z = landmark.z

                # Draw landmark on frame
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Convert frame back to BGR format
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Write frame to file
    frame_path = os.path.join(output_dir, 'frame{}.jpg'.format(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    cv2.imwrite(frame_path, frame)

    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()

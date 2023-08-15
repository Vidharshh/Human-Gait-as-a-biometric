import cv2
import numpy as np
import os
import csv

# Set root directory of GEI images
root_dir = "D:\\College\\6th Sem\\CIP\\Data set\\GEI"

# Set the dimensions of the GEI images
height, width = 128, 88

# Create a list to store the feature vectors and labels
features = []
labels = []

# Loop through each subdirectory in the root directory
for label in os.listdir(root_dir):
    label_dir = os.path.join(root_dir, label)

    # Loop through each GEI image in the label directory
    for gei_file in os.listdir(label_dir):
        # Load the GEI image
        gei_path = os.path.join(label_dir, gei_file)
        gei = cv2.imread(gei_path, cv2.IMREAD_GRAYSCALE)

        # Resize the GEI image to the desired dimensions
        gei = cv2.resize(gei, (width, height))

        # Convert the GEI image to a feature vector
        feature_vector = np.reshape(gei, (height * width))

        # Threshold the feature vector to binary values
        ret, feature_vector = cv2.threshold(feature_vector, 127, 1, cv2.THRESH_BINARY)

        # Append the feature vector and label to the lists
        features.append(feature_vector)
        labels.append(label)

# Convert the lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Save the features and labels to a CSV file
with open("gait_features3.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row
    header = ["label"] + ["pixel_{:04d}".format(i) for i in range(height * width)]
    writer.writerow(header)

    # Write each feature vector and label to a new row in the CSV file
    for i in range(len(features)):
        row = [labels[i]] + list(features[i])
        writer.writerow(row)

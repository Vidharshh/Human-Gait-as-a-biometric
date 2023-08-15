import os
import cv2
import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16, preprocess_input

# Set root directory of GEI images
root_dir = 'D:\\College\\6th Sem\\CIP\\Data set\\GEI_CASIA_B\\gei'

# Set the dimensions of the GEI images
height, width = 128, 88

# Load VGG16 model
vgg16 = VGG16(weights='imagenet', include_top=False)

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

        # Convert the GEI image to RGB
        gei = cv2.cvtColor(gei, cv2.COLOR_GRAY2RGB)

        # Preprocess the GEI image for VGG16 model
        gei = preprocess_input(gei)

        # Extract features from the GEI image using VGG16 model
        features_vector = vgg16.predict(np.expand_dims(gei, axis=0))
        features_vector = np.ravel(features_vector)

        # Append the feature vector and label to the lists
        features.append(features_vector)
        labels.append(label)

# Convert the lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Convert features to dataframe
all_features_df = pd.DataFrame(features)

# Add label column to the dataframe
all_features_df['label'] = labels

# Save the features and labels to a CSV file
all_features_df.to_csv('gait_features_cnn.csv', index=False)
import cv2
import numpy as np
import os
dataset_path = 'D:\\College\\6th Sem\\CIP\\Data set\\GaitDatasetB-silh\\'


def gei(sequence_path):
    # Load all images in the sequence and convert to grayscale
    sequence = []
    for filename in os.listdir(sequence_path):
        if filename.endswith('.png'):
            img_path = os.path.join(sequence_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 64))  # Change (128, 64) to the desired size
            sequence.append(img)

    # Compute the average of all images in the sequence
    gei = np.mean(sequence, axis=0)

    # Normalize the GEI to have a range of [0, 255]
    gei_norm = cv2.normalize(gei, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return gei_norm


def gei_all(dataset_path):
    # Create a folder to store the GEIs
    output_path = 'D:\\College\\6th Sem\\CIP\\Data set\\GEI_CASIA_B\\gei'
    os.makedirs(output_path, exist_ok=True)

    # Loop over all subjects, walks, and trials in the dataset
    for subject in os.listdir(dataset_path):
        if not subject.startswith('.'):
            subject_path = os.path.join(dataset_path, subject)
            for walk in os.listdir(subject_path):
                if not walk.startswith('.'):
                    walk_path = os.path.join(subject_path, walk)
                    for trial in os.listdir(walk_path):
                        if not trial.startswith('.'):
                            trial_path = os.path.join(walk_path, trial)
                            output_filename = '{}_{}_{}.png'.format(subject, walk, trial)
                            output_filepath = os.path.join(output_path, output_filename)
                            gei_image = gei(trial_path)
                            cv2.imwrite(output_filepath, gei_image)
                            print('Processed', output_filename)

gei_all(dataset_path)

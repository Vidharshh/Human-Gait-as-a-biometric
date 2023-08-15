import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load the data
df = pd.read_csv('gait_features3_modified.csv')

# Split the data into training and testing sets
X = df.drop(['label'], axis=1).values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Reshape the data for CNN
X_train_cnn = X_train.reshape(-1, 100, 100, 1)
X_test_cnn = X_test.reshape(-1, 100, 100, 1)

# Build the CNN model
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN model
early_stop = EarlyStopping(monitor='val_loss', patience=3)
cnn_model.fit(X_train_cnn, y_train, validation_data=(X_test_cnn, y_test), epochs=10, callbacks=[early_stop])

# Make predictions using the CNN model
y_pred_cnn_prob = cnn_model.predict(X_test_cnn)
y_pred_cnn = np.argmax(y_pred_cnn_prob, axis=1)

# Encode the labels for XGBoost
label_encoder = LabelEncoder()
y_train_xgb = label_encoder.fit_transform(y_train)
y_test_xgb = label_encoder.transform(y_test)

# Build the XGBoost model
xgb_model = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train_xgb)

# Make predictions using the XGBoost model
y_pred_xgb = xgb_model.predict(X_test)

# Combine the predictions from both models
y_pred_ensemble = []
for i in range(len(y_pred_cnn)):
    if y_pred_cnn[i] == 1:
        y_pred_ensemble.append(label_encoder.inverse_transform([y_pred_xgb[i]])[0])
    else:
        y_pred_ensemble.append('unknown')

# Calculate the accuracy of the ensemble model
accuracy = accuracy_score(y_test, y_pred_ensemble)
print('Accuracy:', accuracy)

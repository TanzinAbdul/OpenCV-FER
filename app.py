import streamlit as st
import cv2
from fer import FER
import pandas as pd
import numpy as np
from datetime import datetime

# Title
st.title("Real-Time Facial Emotion Recognition App")

# Button to start webcam
run = st.checkbox('Start Emotion Detection')

# Placeholder for video frame
FRAME_WINDOW = st.image([])

# Detector
detector = FER(mtcnn=True)

# CSV writer setup
CSV_FILE = "emotion_results.csv"
if run:
    df = pd.DataFrame(columns=["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "detected_emotion", "timestamp"])
    df.to_csv(CSV_FILE, index=False)

# OpenCV video capture
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to access webcam.")
        break

    results = detector.detect_emotions(frame)
    
    for result in results:
        (x, y, w, h) = result["box"]
        emotions = result["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, dominant_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Save to CSV
        new_row = {
            "angry": emotions['angry'], "disgust": emotions['disgust'], "fear": emotions['fear'],
            "happy": emotions['happy'], "sad": emotions['sad'], "surprise": emotions['surprise'],
            "neutral": emotions['neutral'], "detected_emotion": dominant_emotion,
            "timestamp": datetime.now().isoformat()
        }
        pd.DataFrame([new_row]).to_csv(CSV_FILE, mode='a', header=False, index=False)

    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

# Release capture when checkbox is unchecked
cap.release()

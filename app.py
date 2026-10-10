import pickle
from pathlib import Path
import threading

import av
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)

MODEL_PATH = Path("models/model.pkl")

st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Sign Language Detection System")
st.write(
    "Show a trained hand gesture to your camera. "
    "The application predicts the gesture and displays its text label."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found: models/model.pkl")
    st.info(
        "First run `python collect_data.py` to collect samples, "
        "then run `python train_model.py` to train the model."
    )
    st.stop()

# Load the saved model once per Streamlit session.
@st.cache_resource
def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)


model = load_model(str(MODEL_PATH.resolve()))

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.lock = threading.Lock()
        self.prediction = "Waiting for hand..."
        self.confidence = 0.0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                image,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            features = []
            for landmark in hand.landmark:
                features.extend([landmark.x, landmark.y])

            # Keep feature order identical to training.
            with self.lock:
                probabilities = model.predict_proba([features])[0]
                best_index = probabilities.argmax()
                label = str(model.classes_[best_index])
                confidence = float(probabilities[best_index])

                self.prediction = label
                self.confidence = confidence

            text = f"{label} ({confidence:.0%})"
            color = (0, 255, 0)

        else:
            with self.lock:
                self.prediction = "No hand detected"
                self.confidence = 0.0

            text = "No hand detected"
            color = (0, 0, 255)

        cv2.rectangle(image, (0, 0), (image.shape[1], 55), (30, 30, 30), -1)
        cv2.putText(
            image,
            text,
            (15, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        return av.VideoFrame.from_ndarray(image, format="bgr24")


ctx = webrtc_streamer(
    key="sign-language-camera",
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    async_processing=True
)

st.caption(
    "Allow camera access when prompted. Predictions are limited "
    "to the gesture classes included in your training dataset."
)

if ctx.video_processor:
    with ctx.video_processor.lock:
        prediction = ctx.video_processor.prediction
        confidence = ctx.video_processor.confidence

    st.subheader("Detected gesture")
    st.write(prediction)

    if prediction != "No hand detected" and prediction != "Waiting for hand...":
        st.write(f"Model probability for this class: {confidence:.1%}")
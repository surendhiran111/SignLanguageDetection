import base64
import csv
import logging
import os
from pathlib import Path
import pickle
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

MODEL_PATH = Path("models/model.pkl")
DATASET_PATH = Path("dataset/gestures.csv")
FEATURE_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y")]

# Shared locks and state
model_lock = threading.Lock()
mp_lock = threading.Lock()
cv_cam_lock = threading.Lock()

cv_camera = None
is_streaming_cv = False

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
mp_draw = mp.solutions.drawing_utils

# Model container
current_model = None
classes_list = []


def load_model():
    """Load model from disk into memory."""
    global current_model, classes_list
    with model_lock:
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    current_model = pickle.load(f)
                    classes_list = [str(c) for c in current_model.classes_]
                logger.info(f"Model loaded successfully with classes: {classes_list}")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                current_model = None
                classes_list = []
                return False
        else:
            logger.warning("Model file not found at %s", MODEL_PATH)
            current_model = None
            classes_list = []
            return False


# Initial load
load_model()


def get_dataset_stats():
    """Return counts and status of dataset."""
    if not DATASET_PATH.exists():
        return {"total_samples": 0, "classes": {}}
    try:
        df = pd.read_csv(DATASET_PATH)
        if "label" in df.columns:
            counts = df["label"].value_counts().to_dict()
            return {
                "total_samples": int(len(df)),
                "classes": {str(k): int(v) for k, v in counts.items()},
            }
    except Exception as e:
        logger.error(f"Error reading dataset stats: {e}")
    return {"total_samples": 0, "classes": {}}


# Predefined gesture helper metadata
GESTURE_METADATA = {
    "Hi": {
        "title": "Hello / Wave",
        "description": "Open hand with spread fingers facing the camera.",
        "icon": "👋",
        "category": "Greeting",
    },
    "GoodJob": {
        "title": "Thumbs Up",
        "description": "Fist closed with thumb pointing directly upwards.",
        "icon": "👍",
        "category": "Affirmation",
    },
    "Nice": {
        "title": "OK / Pinch",
        "description": "Index finger and thumb touching in a ring, other three fingers extended.",
        "icon": "👌",
        "category": "Feedback",
    },
    "peace": {
        "title": "Peace / Victory",
        "description": "Index and middle fingers extended upward in a 'V' shape, others curled.",
        "icon": "✌️",
        "category": "Expression",
    },
}


@app.route("/")
def index():
    """Render the primary single-page application."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """Return system readiness, loaded model, and dataset statistics."""
    stats = get_dataset_stats()
    with model_lock:
        has_model = current_model is not None
        cls = list(classes_list)

    return jsonify(
        {
            "model_loaded": has_model,
            "classes": cls,
            "dataset_stats": stats,
            "gesture_metadata": GESTURE_METADATA,
            "server_time": time.time(),
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Process an image frame from the browser webcam and return gesture predictions."""
    start_time = time.perf_counter()

    data = request.get_json(silent=True) or {}
    image_data = data.get("image")

    if not image_data:
        return jsonify({"success": False, "error": "No image data provided"}), 400

    try:
        # Extract base64 payload
        if "," in image_data:
            _, encoded = image_data.split(",", 1)
        else:
            encoded = image_data

        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "error": "Failed to decode image"}), 400

        # Run MediaPipe Hands detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with mp_lock:
            result = hands_detector.process(rgb)

        if not result.multi_hand_landmarks:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return jsonify(
                {
                    "success": True,
                    "hand_detected": False,
                    "prediction": "No hand detected",
                    "confidence": 0.0,
                    "probabilities": {c: 0.0 for c in classes_list},
                    "landmarks": [],
                    "inference_ms": round(elapsed_ms, 2),
                }
            )

        hand = result.multi_hand_landmarks[0]

        # Extract 21 landmarks (x, y)
        features = []
        landmarks_list = []
        for lm in hand.landmark:
            features.extend([lm.x, lm.y])
            landmarks_list.append({"x": lm.x, "y": lm.y, "z": lm.z})

        # Run model inference if loaded
        prediction = "Waiting for model..."
        confidence = 0.0
        probabilities_dict = {}

        with model_lock:
            if current_model is not None:
                features_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
                probabilities = current_model.predict_proba(features_df)[0]
                best_idx = probabilities.argmax()
                prediction = str(current_model.classes_[best_idx])
                confidence = float(probabilities[best_idx])
                probabilities_dict = {
                    str(c): float(p) for c, p in zip(current_model.classes_, probabilities)
                }
            else:
                prediction = "Model not trained yet"

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return jsonify(
            {
                "success": True,
                "hand_detected": True,
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "probabilities": probabilities_dict,
                "landmarks": landmarks_list,
                "inference_ms": round(elapsed_ms, 2),
            }
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/save_sample", methods=["POST"])
def api_save_sample():
    """Save landmark coordinates directly from the web client into dataset/gestures.csv."""
    data = request.get_json(silent=True) or {}
    label = (data.get("label") or "").strip()
    landmarks = data.get("landmarks") or []

    if not label:
        return jsonify({"success": False, "error": "Gesture label is required"}), 400

    if len(landmarks) != 21:
        return jsonify({"success": False, "error": "Expected 21 hand landmarks"}), 400

    try:
        features = []
        for lm in landmarks:
            features.extend([lm.get("x", 0.0), lm.get("y", 0.0)])

        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = not DATASET_PATH.exists() or DATASET_PATH.stat().st_size == 0

        with open(DATASET_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(FEATURE_COLUMNS + ["label"])
            writer.writerow(features + [label])

        return jsonify({"success": True, "message": f"Sample saved for gesture '{label}'"})
    except Exception as e:
        logger.error(f"Error saving sample: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    """Trigger model training from dataset/gestures.csv and reload model in memory."""
    if not DATASET_PATH.exists():
        return jsonify({"success": False, "error": "Dataset file gestures.csv not found"}), 400

    try:
        df = pd.read_csv(DATASET_PATH)
        required = FEATURE_COLUMNS + ["label"]
        missing = set(required) - set(df.columns)
        if missing:
            return jsonify({"success": False, "error": f"Missing columns: {list(missing)}"}), 400

        df = df[required].copy()
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=required)
        df["label"] = df["label"].astype(str).str.strip()
        df = df[df["label"] != ""]

        counts = df["label"].value_counts()
        if len(counts) < 2 or (counts < 2).any():
            return jsonify(
                {
                    "success": False,
                    "error": "Dataset must contain at least 2 distinct classes with >=2 samples each.",
                }
            ), 400

        X = df[FEATURE_COLUMNS]
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
        )
        model.fit(X_train, y_train)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        # Reload in memory
        load_model()

        acc = float(model.score(X_test, y_test))
        return jsonify(
            {
                "success": True,
                "accuracy": round(acc, 4),
                "classes": list(model.classes_),
                "total_samples": int(len(df)),
            }
        )
    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def generate_cv_frames():
    """MJPEG stream generator for server-side OpenCV camera."""
    global cv_camera, is_streaming_cv
    with cv_cam_lock:
        if cv_camera is None or not cv_camera.isOpened():
            cv_camera = cv2.VideoCapture(0)

    try:
        while is_streaming_cv:
            success, frame = cv_camera.read()
            if not success:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with mp_lock:
                result = hands_detector.process(rgb)

            label_text = "No hand detected"
            color = (0, 0, 255)

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                features = []
                for lm in hand.landmark:
                    features.extend([lm.x, lm.y])

                with model_lock:
                    if current_model is not None:
                        features_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
                        probs = current_model.predict_proba(features_df)[0]
                        best = probs.argmax()
                        label = str(current_model.classes_[best])
                        conf = float(probs[best])
                        label_text = f"{label} ({conf:.0%})"
                        color = (0, 255, 0)

            # Draw HUD bar
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (20, 20, 20), -1)
            cv2.putText(
                frame,
                f"Gesture: {label_text}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(0.03)
    finally:
        with cv_cam_lock:
            if cv_camera is not None and cv_camera.isOpened():
                cv_camera.release()
                cv_camera = None


@app.route("/video_feed")
def video_feed():
    """Route for MJPEG camera stream."""
    global is_streaming_cv
    is_streaming_cv = True
    return Response(
        generate_cv_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/stop_video_feed", methods=["POST"])
def stop_video_feed():
    """Stop server-side OpenCV capture to free hardware camera resources."""
    global is_streaming_cv, cv_camera
    is_streaming_cv = False
    with cv_cam_lock:
        if cv_camera is not None and cv_camera.isOpened():
            cv_camera.release()
            cv_camera = None
    return jsonify({"success": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n==================================================")
    print(f"🚀 Sign Language Detection Web Server")
    print(f"📡 Serving on http://localhost:{port}")
    print(f"==================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
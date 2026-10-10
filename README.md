# 🤟 Real-Time Sign Language Detection Studio

An interactive machine learning web application for real-time hand gesture and sign language recognition using **MediaPipe**, **OpenCV**, **Scikit-Learn**, and a modern **HTML5 / CSS / JavaScript** frontend powered by a high-performance **Flask** backend.

---

## 📌 Features

- **Real-Time Hand Landmark Tracking**: Utilizes Google MediaPipe Hands to extract 21 2D landmark coordinates (42 features per frame).
- **High-Accuracy Classification**: Powered by a trained Random Forest classifier predicting gesture classes and confidence probability distribution.
- **Modern Web Frontend**:
  - Built with clean semantic **HTML5**, **Vanilla JavaScript**, and modular **Vanilla CSS in a dedicated `static/css/` directory**.
  - **Zero Streamlit Dependency**: High framerate, zero-lag browser camera capture via WebRTC / `navigator.mediaDevices`.
  - **Holographic Landmark Overlay**: Real-time canvas rendering of glowing hand skeleton joints and connections.
  - **Sign-to-Speech Sentence Builder**: Natural text-to-speech (TTS) translation engine accumulating gestures into spoken sentences.
  - **Telemetry & Ops Dashboard**: Live FPS meter, inference latency monitor, multiclass softmax probability bars, and 1-click model retraining.
- **Custom Gesture Dataset Collection**: Includes a built-in CLI tool (`collect_data.py`) to record custom hand gestures directly from your webcam.

---

## 📁 Project Structure

```text
SIGNLANGUAGE/
├── app.py                  # High-performance Flask web server & prediction API
├── collect_data.py         # CLI tool to capture gesture landmarks from webcam
├── train_model.py          # Random Forest model training script
├── requirements.txt        # Python dependencies (Flask, MediaPipe, OpenCV, Scikit-Learn)
├── dataset/                # Stores collected gesture landmark CSV data
│   └── gestures.csv        # 800 samples across 4 classes
├── models/                 # Stores trained model pickle file (model.pkl)
│   └── model.pkl
├── static/                 # Static assets folder
│   ├── css/                # Dedicated separate CSS folder
│   │   ├── style.css       # Design tokens, typography, glassmorphism, reset & layout
│   │   ├── components.css  # Video viewport, HUD badges, prediction cards, gauges
│   │   └── animations.css  # Scanlines, glowing radar blips, pulse effects
│   └── js/
│       └── app.js          # WebRTC camera, canvas skeleton renderer, TTS, prediction loop
├── templates/
│   └── index.html          # Modern Single-Page Application UI
├── .gitignore              # Git ignore configuration
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+ installed on your machine.
- A functional webcam.

### 2. Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/surendhiran111/SignLanguageDetection.git
cd SignLanguageDetection

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🛠️ How to Run

### Run the Web Application

Start the web server:

```bash
python app.py
```

Then open your browser at:
**[http://localhost:5001](http://localhost:5001)**

1. Click **"Start Camera"** to activate the webcam.
2. Allow browser camera permissions when prompted.
3. Show any trained hand gesture (`Hi`, `GoodJob`, `Nice`, `peace`).
4. Watch the real-time glowing skeleton, confidence gauge, class probabilities, and hear the speech translation!

---

## 🖐️ Adding New Gestures

### Step 1: Collect Custom Gesture Data

```bash
python collect_data.py
```

1. Enter the gesture name when prompted (e.g., `Hello`, `ThankYou`, `Peace`, `Yes`, `No`).
2. Point your hand towards the webcam.
3. Press **`S`** to record 200 landmark samples.
4. Press **`Q`** to finish.
5. Repeat for each custom gesture.

### Step 2: Retrain the Model

You can either:
- Click the **"⚡ Retrain Model"** button directly inside the web interface, or
- Run via terminal:
  ```bash
  python train_model.py
  ```

---

## 📦 Dependencies

- `Flask`
- `flask-cors`
- `mediapipe`
- `opencv-python`
- `scikit-learn`
- `pandas`
- `numpy`

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

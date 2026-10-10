# 🤟 Real-Time Sign Language Detection System

An interactive machine learning application for real-time hand gesture and sign language recognition using **MediaPipe**, **OpenCV**, **Scikit-Learn**, and **Streamlit**.

---

## 📌 Features

- **Real-Time Hand Landmark Tracking**: Utilizes Google MediaPipe Hands to extract 21 2D landmark coordinates (42 features per frame).
- **Fast & Accurate Classification**: Trained with a Random Forest classifier for high accuracy and fast real-time inference.
- **Interactive Web App**: Powered by Streamlit and `streamlit-webrtc` for real-time browser-based video streaming and live gesture prediction.
- **Custom Gesture Dataset Collection**: Includes a built-in CLI tool to record custom hand gestures directly from your webcam.

---

## 📁 Project Structure

```text
SIGNLANGUAGE/
├── app.py                # Streamlit web application with live webcam inference
├── collect_data.py       # Script to capture gesture landmarks from webcam
├── train_model.py        # Random Forest model training script
├── requirements.txt      # Project dependencies
├── dataset/              # Stores collected gesture landmark CSV data
│   └── .gitkeep
├── models/               # Stores trained model pickle file (model.pkl)
│   └── .gitkeep
├── .gitignore            # Git ignore configuration
└── README.md             # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.9+ installed on your machine.

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

## 🛠️ Workflow

### Step 1: Collect Gesture Data

Run the collection script to record landmark coordinates for different gestures:

```bash
python collect_data.py
```

1. Enter the gesture name when prompted (e.g., `Hello`, `ThankYou`, `Peace`, `Yes`, `No`).
2. Point your hand towards the webcam.
3. Press **`S`** to start collecting 200 landmark samples.
4. Press **`Q`** to quit anytime.
5. Repeat for each gesture you wish to recognize.

Data will be stored in `dataset/gestures.csv`.

---

### Step 2: Train the Classifier

Once you have recorded at least 2 distinct gestures:

```bash
python train_model.py
```

This will:
- Clean and normalize landmark coordinates.
- Train a Random Forest classifier with stratified train/test split.
- Display accuracy, confusion matrix, and classification report.
- Save the trained model to `models/model.pkl`.

---

### Step 3: Run the Streamlit Application

Start the web application:

```bash
streamlit run app.py
```

- Open the local URL displayed in your terminal (usually `http://localhost:8501`).
- Allow webcam permissions.
- Perform any trained gesture in front of the camera to see real-time predictions and confidence scores!

---

## 📦 Dependencies

- `mediapipe`
- `opencv-python`
- `scikit-learn`
- `pandas`
- `numpy`
- `streamlit`
- `streamlit-webrtc`
- `av`

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

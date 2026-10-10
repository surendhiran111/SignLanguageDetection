import csv
import time
import cv2
import mediapipe as mp
from pathlib import Path

DATASET_PATH = Path("dataset/gestures.csv")
DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

TARGET_SAMPLES = 200

FEATURE_COLUMNS = [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y")
]
CSV_COLUMNS = FEATURE_COLUMNS + ["label"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

gesture = input("Enter gesture label (e.g., Hi): ").strip()

if not gesture:
    raise SystemExit("Gesture label cannot be empty.")

# Create CSV header if needed.
if not DATASET_PATH.exists() or DATASET_PATH.stat().st_size == 0:
    with open(DATASET_PATH, "w", newline="") as file:
        csv.writer(file).writerow(CSV_COLUMNS)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("Cannot open webcam. Check camera permissions.")

saved_count = 0
capturing = False

try:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands, open(DATASET_PATH, "a", newline="") as file:

        writer = csv.writer(file)

        print("Show your gesture.")
        print("Press S to start capturing 200 samples.")
        print("Press Q to quit.")

        while True:
            success, frame = cap.read()

            if not success:
                print("Failed to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            features = None

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

                features = []
                for point in hand.landmark:
                    features.extend([point.x, point.y])

            # Automatically save one valid sample per frame.
            if capturing and features is not None:
                writer.writerow(features + [gesture])
                saved_count += 1

                # Flush writes so collected data is saved promptly.
                if saved_count % 10 == 0:
                    file.flush()

                if saved_count >= TARGET_SAMPLES:
                    file.flush()
                    capturing = False
                    print(f"\nCompleted: {saved_count} samples for {gesture}")

            status = (
                f"CAPTURING: {saved_count}/{TARGET_SAMPLES}"
                if capturing else
                f"READY: {saved_count}/{TARGET_SAMPLES}"
            )

            cv2.putText(
                frame, f"Gesture: {gesture}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2
            )
            cv2.putText(
                frame, status,
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0) if capturing else (0, 200, 255), 2
            )
            cv2.putText(
                frame, "S: Start/Resume | R: Reset count | Q: Quit",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

            cv2.imshow("Gesture Data Collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s") and not capturing:
                if saved_count >= TARGET_SAMPLES:
                    print("Target already reached. Restart for another gesture.")
                else:
                    capturing = True
                    print("Capturing samples... Keep your hand visible.")

            elif key == ord("r") and not capturing:
                print("Count reset. Existing CSV rows are not deleted.")
                saved_count = 0

            elif key == ord("q") or key == 27:
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

print(f"Session ended. Samples collected this session: {saved_count}")
print(f"Dataset location: {DATASET_PATH.resolve()}")
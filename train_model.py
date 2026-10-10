from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATASET_PATH = Path("dataset/gestures.csv")
MODEL_PATH = Path("models/model.pkl")

FEATURE_COLUMNS = [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y")
]


def main():
    if not DATASET_PATH.exists():
        raise SystemExit(
            "Dataset not found. Run python collect_data.py first."
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = FEATURE_COLUMNS + ["label"]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise SystemExit(f"Missing CSV columns: {sorted(missing)}")

    df = df[required_columns].copy()

    # Validate and clean rows.
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df.dropna(subset=required_columns)

    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"] != ""]

    if df.empty:
        raise SystemExit("No valid training samples found.")

    counts = df["label"].value_counts()

    print("\nSamples per gesture:")
    print(counts)

    if len(counts) < 2:
        raise SystemExit(
            "Collect at least two different gesture classes."
        )

    if (counts < 2).any():
        raise SystemExit(
            "Each gesture needs at least two samples. "
            "Collect more data and retry."
        )

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    # Reserve data that is not used to fit the model.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=model.classes_
        )
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print(f"\nModel saved successfully: {MODEL_PATH.resolve()}")
    print(f"Number of training features: {len(FEATURE_COLUMNS)}")
    print(f"Gesture classes: {list(model.classes_)}")


if __name__ == "__main__":
    main()
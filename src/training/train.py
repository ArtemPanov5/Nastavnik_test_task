from __future__ import annotations
import sys
import json
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from data.data_preprocessing import engineer_features, load_interactions, print_basic_stats

FEATURE_COLUMNS = ["num_attempts", "success_rate", "last_correct", "learning_curve"]


def train_model(features_df):
    df = features_df.copy()
    df["target"] = (df["success_rate"] > 0.75).astype(int)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=int)

    # проверить сколько данных
    if len(X) < 10:
        print("Warning: Dataset too small for train/test split. Training on full dataset.")
        model = LogisticRegression(solver="liblinear", random_state=42)
        model.fit(X, y)

        probs = model.predict_proba(X)[:, 1]
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(y, preds)

        print(f"Train accuracy (full dataset): {acc:.4f}")
        return model, acc, None

    # Train/test split (80/20) нужно так-то больше данных
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback: split без stratify баланс классов
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_train, y_train)

    # Train metrics
    train_probs = model.predict_proba(X_train)[:, 1]
    train_preds = (train_probs > 0.5).astype(int)
    train_acc = accuracy_score(y_train, train_preds)

    # Test metrics
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs > 0.5).astype(int)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    return model, train_acc, test_acc


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw_data" / "interactions.json"
    preprocessed_dir = project_root / "data" / "preprocessed_data"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading interactions from: {data_path}")
    interactions = load_interactions(data_path)
    print_basic_stats(interactions)

    features = engineer_features(interactions)
    print(f"Features shape: {features.shape}")
    print(features.head(10).to_string(index=False))

    features_csv = preprocessed_dir / "features.csv"
    interactions_csv = preprocessed_dir / "interactions_clean.csv"

    features.to_csv(features_csv, index=False)
    interactions.to_csv(interactions_csv, index=False)

    print(f"Saved features to: {features_csv}")
    print(f"Saved cleaned interactions to: {interactions_csv}")

    result = train_model(features)
    model = result[0]

    model_path = models_dir / "knowledge_model.joblib"
    cols_path = models_dir / "feature_columns.json"

    joblib.dump(model, model_path)
    cols_path.write_text(json.dumps(FEATURE_COLUMNS, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved model to: {model_path}")
    print(f"Saved feature columns to: {cols_path}")


if __name__ == "__main__":
    main()
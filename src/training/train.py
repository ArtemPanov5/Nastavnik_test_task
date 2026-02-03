from __future__ import annotations
import sys
import json
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from data.data_preprocessing import engineer_features, load_interactions, print_basic_stats

FEATURE_COLUMNS = ["num_attempts", "success_rate", "last_correct", "learning_curve"]

def train_model(features_df):
    df = features_df.copy()
    df["target"] = (df["success_rate"] > 0.75).astype(int)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=int)

    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y, preds)

    print(f"Train accuracy: {acc:.4f}")
    return model, acc


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw_data" / "interactions.json"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading interactions from: {data_path}")
    interactions = load_interactions(data_path)
    print_basic_stats(interactions)

    features = engineer_features(interactions)
    print(f"Features shape: {features.shape}")
    print(features.head(10).to_string(index=False))

    features_csv = project_root / "data" / "preprocessed_data" / "features.csv"
    interactions_csv = project_root / "data" / "preprocessed_data" / "interactions_clean.csv"

    features.to_csv(features_csv, index=False)
    interactions.to_csv(interactions_csv, index=False)

    print(f"Saved features to: {features_csv}")
    print(f"Saved cleaned interactions to: {interactions_csv}")

    model, _ = train_model(features)

    model_path = models_dir / "knowledge_model.joblib"
    cols_path = models_dir / "feature_columns.json"

    joblib.dump(model, model_path)
    cols_path.write_text(json.dumps(FEATURE_COLUMNS, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved model to: {model_path}")
    print(f"Saved feature columns to: {cols_path}")


if __name__ == "__main__":
    main()
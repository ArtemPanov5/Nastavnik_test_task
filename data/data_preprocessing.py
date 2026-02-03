from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["user_id", "skill_id", "correctness", "attempt_num"]


def load_interactions(filepath: str | Path) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Interactions file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw)
    except Exception as e:
        raise ValueError(f"Failed to read JSON from {path}: {e}") from e

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Got columns={list(df.columns)}")

    df = df[REQUIRED_COLUMNS].copy()

    df["user_id"] = df["user_id"].astype(str)
    df["skill_id"] = df["skill_id"].astype(str)
    df["correctness"] = pd.to_numeric(df["correctness"], errors="coerce")
    df["attempt_num"] = pd.to_numeric(df["attempt_num"], errors="coerce")

    if df["correctness"].isna().any() or df["attempt_num"].isna().any():
        bad_rows = df[df["correctness"].isna() | df["attempt_num"].isna()]
        raise ValueError(f"Found non-numeric correctness/attempt_num rows:\n{bad_rows.head(10)}")

    df["correctness"] = df["correctness"].astype(int)
    df["attempt_num"] = df["attempt_num"].astype(int)

    if not df["correctness"].isin([0, 1]).all():
        bad = df.loc[~df["correctness"].isin([0, 1]), ["user_id", "skill_id", "correctness", "attempt_num"]]
        raise ValueError(f"correctness must be 0/1. Bad rows:\n{bad.head(10)}")

    return df


def print_basic_stats(df: pd.DataFrame) -> None:
    n_users = df["user_id"].nunique()
    n_skills = df["skill_id"].nunique()
    avg_correct = float(df["correctness"].mean()) if len(df) else 0.0

    print(f"Unique users: {n_users}")
    print(f"Unique skills: {n_skills}")
    print(f"Average correctness: {avg_correct:.4f}")


def _learning_curve(correctness_series: pd.Series) -> float:
    tail = correctness_series.tail(3)
    return float(tail.mean()) if len(tail) else 0.0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["user_id", "skill_id", "num_attempts", "success_rate", "last_correct", "learning_curve"]
        )

    df_sorted = df.sort_values(["user_id", "skill_id", "attempt_num"]).copy()

    grouped = df_sorted.groupby(["user_id", "skill_id"], as_index=False)

    feats = grouped.agg(
        num_attempts=("correctness", "size"),
        success_rate=("correctness", "mean"),
        last_correct=("correctness", "last"),
    )

    lc = (
        df_sorted.groupby(["user_id", "skill_id"])["correctness"]
        .apply(_learning_curve)
        .reset_index(name="learning_curve")
    )

    out = feats.merge(lc, on=["user_id", "skill_id"], how="left")

    out["num_attempts"] = out["num_attempts"].astype(int)
    out["success_rate"] = out["success_rate"].astype(float)
    out["last_correct"] = out["last_correct"].astype(int)
    out["learning_curve"] = out["learning_curve"].astype(float)

    return out
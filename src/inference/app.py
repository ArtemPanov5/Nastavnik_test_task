from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, confloat, conint

logger = logging.getLogger("nastavnik")
logging.basicConfig(level=logging.INFO)


class StudentQuery(BaseModel):
    user_id: str
    skill_id: str
    num_attempts: conint(ge=0)
    success_rate: confloat(ge=0.0, le=1.0)
    last_correct: conint(ge=0, le=1)
    learning_curve: confloat(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    knows_skill: bool
    confidence: confloat(ge=0.0, le=1.0)

app = FastAPI(title="Nastavnik Knowledge Prediction")

MODEL = None
FEATURE_COLUMNS: Optional[list[str]] = None


def _artifact_paths():
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"
    return models_dir / "knowledge_model.joblib", models_dir / "feature_columns.json"


@app.on_event("startup")
def load_model_on_startup():
    global MODEL, FEATURE_COLUMNS
    model_path, cols_path = _artifact_paths()
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not cols_path.exists():
            raise FileNotFoundError(f"Feature columns file not found: {cols_path}")
        MODEL = joblib.load(model_path)
        FEATURE_COLUMNS = json.loads(cols_path.read_text(encoding="utf-8"))
        logger.info("Model loaded from %s", model_path)
        logger.info("Feature columns: %s", FEATURE_COLUMNS)
    except Exception as e:
        MODEL = None
        FEATURE_COLUMNS = None
        logger.exception("Failed to load model artifacts: %s", e)


@app.post("/predict_knowledge", response_model=PredictionResponse)
async def predict_knowledge(query: StudentQuery):
    """
    Предсказать, знает ли студент навык.
    knows_skill: True если модель выдала > 0.5
    confidence: сырое предсказание модели (вероятность от 0 до 1)
    """
    try:
        if MODEL is None or not FEATURE_COLUMNS:
            raise HTTPException(status_code=503, detail="Model is not loaded")

        payload = query.model_dump()
        x = np.array([[payload[col] for col in FEATURE_COLUMNS]], dtype=float)
        proba = float(MODEL.predict_proba(x)[0, 1])
        knows = bool(proba > 0.5)

        logger.info(
            "Predict user_id=%s skill_id=%s proba=%.4f",
            query.user_id,
            query.skill_id,
            proba,
        )

        return PredictionResponse(knows_skill=knows, confidence=proba)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/health")
def health_check():
    ready = MODEL is not None and FEATURE_COLUMNS is not None
    return {"status": "ok", "model_ready": ready}
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "category_model.joblib"
THRESHOLD = 0.5          # is se kam confidence = "no suggestion"
bundle = None            # {"tfidf": ..., "model": ...}


# ---------- Schemas ----------
class ComplaintIn(BaseModel):
    title: str = Field("", max_length=300)
    description: str = Field("", max_length=3000)

class Suggestion(BaseModel):
    category: str
    confidence: float

class PredictionOut(BaseModel):
    category: Optional[str]        # None = koi suggestion nahi
    confidence: float
    low_confidence: bool
    suggestions: List[Suggestion]
    ml_available: bool


# ---------- Model loading ----------
def load_model():
    global bundle
    if bundle is None:
        bundle = joblib.load(MODEL_PATH)
    return bundle

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
        print("Model loaded")
    except Exception as e:
        print("Model load failed:", e)
    yield


# ---------- App ----------
app = FastAPI(title="CampusPulse ML Service", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],   # apna React/Vite URL
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    try:
        load_model()
        return {"status": "ok", "model_loaded": True}
    except Exception:
        return {"status": "degraded", "model_loaded": False}


@app.post("/predict", response_model=PredictionOut)
def predict(c: ComplaintIn):
    if not (c.title.strip() or c.description.strip()):
        raise HTTPException(status_code=400, detail="title or description required")

    try:
        b = load_model()
        text = f"{c.title} {c.description}".strip()
        X = b["tfidf"].transform([text])
        proba = b["model"].predict_proba(X)[0]
        classes = b["model"].classes_

        order = proba.argsort()[::-1][:3]
        top = [{"category": str(classes[i]), "confidence": round(float(proba[i]), 4)} for i in order]
        best = top[0]
        low = best["confidence"] < THRESHOLD

        return {
            "category": None if low else best["category"],
            "confidence": best["confidence"],
            "low_confidence": low,
            "suggestions": top,
            "ml_available": True,
        }
    except Exception:
        # Fallback: ML fail ho jaye tab bhi complaint report hona chahiye
        return {"category": None, "confidence": 0.0, "low_confidence": True,
                "suggestions": [], "ml_available": False}
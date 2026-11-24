"""
CareerShield Backend — serves the EMSCAD-trained ML model.

Loads fraud_model.pkl + tfidf_vectorizer.pkl (from the Kaggle training script)
and exposes a single /analyze endpoint that the React frontend calls.
"""

import re
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="CareerShield ML Backend")

# Allow the local frontend (Vite dev server) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # fine for local demo; restrict in real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Load the trained model + vectorizer (must be in this folder)
# ------------------------------------------------------------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    model = joblib.load(os.path.join(BASE_DIR, "fraud_model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
except FileNotFoundError as e:
    raise SystemExit(
        f"\n{e}\nPut fraud_model.pkl and tfidf_vectorizer.pkl (downloaded from your Kaggle notebook) "
        "in the same folder as main.py, then start the server again.\n"
    )


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ListingText(BaseModel):
    text: str


class LogEntry(BaseModel):
    company: str = "Unknown"
    role: str = "Not specified"
    score: float
    source: str = "Chrome Extension"
    date: str = ""
    tags: list[str] = []


# Simple in-memory log shared between the web app and the Chrome extension.
# Resets when the server restarts — fine for a demo/viva; a real deployment
# would use MongoDB Atlas as per the synopsis.
submissions_log = []


@app.get("/")
def health():
    return {"status": "CareerShield ML backend is running"}


@app.post("/analyze")
def analyze(payload: ListingText):
    cleaned = clean_text(payload.text)
    if not cleaned:
        return {"fraud_probability": 0.0, "model": "Random Forest (TF-IDF, trained on EMSCAD)"}
    vec = vectorizer.transform([cleaned])
    probability = model.predict_proba(vec)[0][1]  # probability of "fraudulent"
    return {
        "fraud_probability": round(float(probability) * 100, 2),
        "model": "Random Forest (TF-IDF, trained on EMSCAD)",
    }


@app.post("/log")
def log_submission(entry: LogEntry):
    data = entry.model_dump() if hasattr(entry, "model_dump") else entry.dict()
    submissions_log.insert(0, data)
    return {"ok": True, "count": len(submissions_log)}


@app.get("/submissions")
def get_submissions():
    return {"submissions": submissions_log}
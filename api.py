"""
Web UI + HTTP API.
Run:  MODEL_DIR=models/v1 API_KEY=secret uvicorn api:app --port 8000
Then open http://127.0.0.1:8000/
"""
import os
import threading
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from spamfilter.model import SpamFilter
from spamfilter.store import Store
from spamfilter.training import FEEDBACK_WEIGHT, train_model

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/v1"))
MODELS_ROOT = MODEL_DIR.parent
DATA_FILE = Path(os.getenv("DATA_FILE", "data/emails.csv"))
FEEDBACK_FILE = Path(os.getenv("FEEDBACK_FILE", "data/feedback.csv"))
API_KEY = os.getenv("API_KEY")  # leave unset only for local testing
UI_FILE = Path(__file__).parent / "static" / "index.html"

app = FastAPI(title="Spam Filter Service")
store = Store(FEEDBACK_FILE, MODELS_ROOT)
model = SpamFilter.load(str(store.active_dir(MODEL_DIR)))
_retrain_lock = threading.Lock()

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)  # gives /docs an Authorize button


def _auth(key: str = Depends(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


class Email(BaseModel):
    subject: str = Field("", max_length=1000)
    body: str = Field("", max_length=200_000)
    sender: str = ""


class Feedback(Email):
    correct_label: str = Field(..., pattern="^(spam|ham)$")


class ActivateRequest(BaseModel):
    name: str


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(UI_FILE)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": model.meta["version"], "thresholds": model.thresholds,
            "auth_required": bool(API_KEY)}


@app.post("/predict", dependencies=[Depends(_auth)])
def predict(email: Email):
    return model.predict(email.subject, email.body, email.sender)


@app.post("/feedback", dependencies=[Depends(_auth)])
def feedback(fb: Feedback):
    """People report mistakes (or confirm correct results). These become training data."""
    store.add_feedback(fb.subject, fb.body, fb.correct_label)
    return {"status": "recorded"}


@app.get("/feedback/stats", dependencies=[Depends(_auth)])
def feedback_stats():
    stats = store.feedback_stats(used=model.meta.get("feedback_rows_used", 0))
    return {**stats, "feedback_weight": FEEDBACK_WEIGHT}


@app.post("/retrain", dependencies=[Depends(_auth)])
def retrain():
    """Train a NEW version on base data + feedback and switch to it. Old versions are kept."""
    global model
    if not _retrain_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A retraining is already running")
    try:
        version = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = MODELS_ROOT / version
        try:
            meta = train_model(DATA_FILE, FEEDBACK_FILE, str(out), version=version)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        model = SpamFilter.load(str(out))
        store.set_active(version)
        return meta
    finally:
        _retrain_lock.release()


@app.get("/models", dependencies=[Depends(_auth)])
def list_models():
    return store.list_models(model.meta["version"])


@app.post("/models/activate", dependencies=[Depends(_auth)])
def activate(req: ActivateRequest):
    """Roll back (or forward) to any saved model version."""
    global model
    d = store.resolve(req.name)
    if d is None:
        raise HTTPException(status_code=404, detail="model version not found")
    model = SpamFilter.load(str(d))
    store.set_active(d.name)
    return {"status": "activated", "model_version": model.meta["version"]}
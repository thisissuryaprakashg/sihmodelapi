import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

# Candidate hazard labels (same as your original code)
HAZARD_LABELS = ["Flood", "Tsunami", "High Waves", "Cyclone", "Noise"]

# Hugging Face model
HF_MODEL = "joeddav/xlm-roberta-large-xnli"
HF_TOKEN = os.environ.get("HF_TOKEN")  # keep token secret

app = FastAPI(title="Hazard Tweet Classifier API")

class TweetRequest(BaseModel):
    tweet: str

class TweetResponse(BaseModel):
    labels: list[str]
    scores: list[float]

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/classify", response_model=TweetResponse)
def classify(req: TweetRequest):
    if not req.tweet.strip():
        return {"labels": [], "scores": []}

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": req.tweet,
        "parameters": {"candidate_labels": HAZARD_LABELS}
    }

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers,
        json=payload
    )

    data = response.json()

    # Handle HF API errors
    if "error" in data:
        return {"labels": [], "scores": []}

    return {"labels": data["labels"], "scores": data["scores"]}

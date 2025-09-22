import warnings
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Suppress FutureWarnings and HF download warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Candidate hazard labels
HAZARD_LABELS = ["Flood", "Tsunami", "High Waves", "Cyclone", "Noise"]

app = FastAPI(title="Hazard Tweet Classifier API")

# Model placeholder
classifier = None

class TweetRequest(BaseModel):
    tweet: str

class TweetResponse(BaseModel):
    labels: list[str]
    scores: list[float]

def get_classifier():
    global classifier
    if classifier is None:
        # Initialize pipeline with slow tokenizer to avoid SentencePiece issues
        classifier = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
            device=-1,       # CPU only
            tokenizer="joeddav/xlm-roberta-large-xnli",  # ensures slow tokenizer
            framework="pt"   # PyTorch backend
        )
    return classifier
@app.on_event("startup")
def load_model():
    get_classifier()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/classify", response_model=TweetResponse)
def classify(req: TweetRequest):
    if not req.tweet.strip():
        return {"labels": [], "scores": []}
    
    clf = get_classifier()
    result = clf(req.tweet, candidate_labels=HAZARD_LABELS)
    return {"labels": result["labels"], "scores": result["scores"]}

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Load Model & Vectorizer
# ------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ------------------------
# Request Schema
# ------------------------
class NewsRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(request: NewsRequest):
    try:
        text_vector = vectorizer.transform([request.text])

        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0][prediction]

        result = "FAKE" if prediction == 0 else "REAL"

        return {
            "prediction": result,
            "confidence": round(float(probability) * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}

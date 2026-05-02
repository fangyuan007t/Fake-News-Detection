from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from lime.lime_text import LimeTextExplainer
import joblib

import numpy as np 
import shap

from PIL import Image
import pytesseract
import io



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Load Model & Vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# SHAP Explainer (for Logistic Regression)
background = vectorizer.transform([
    "news example one",
    "news example two"
]).toarray()

explainer = shap.LinearExplainer(model, background)

class_names = ["FAKE", "REAL"]
lime_explainer = LimeTextExplainer(class_names=class_names)

def predict_proba_lime(texts):
    vec = vectorizer.transform(texts)
    return model.predict_proba(vec)

# Request Schema
class NewsRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(request: NewsRequest):
    try:
        text = request.text

        text_vector = vectorizer.transform([text])

        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0][prediction]

        result = "FAKE" if prediction == 0 else "REAL"

        # ===== SHAP (correct mapping) =====
        text_vector_dense = text_vector.toarray()
        shap_values = explainer(text_vector_dense)

        feature_names = vectorizer.get_feature_names_out()
        shap_dict = dict(zip(feature_names, shap_values.values[0]))

        # ===== LIME (for clean highlighting) =====
        lime_exp = lime_explainer.explain_instance(
            text,
            predict_proba_lime,
            num_features=10
        )

        lime_dict = dict(lime_exp.as_list())

        # ===== Final explanation (use LIME for UI) =====
        words = text.split()

        explanation = [
            {
                "word": w,
                "importance": float(lime_dict.get(w, 0))
            }
            for w in words
        ]

        return {
            "prediction": result,
            "confidence": round(float(probability) * 100, 2),

            # 👇 frontend uses this
            "explanation": explanation,

            # 👇 optional (for viva / graphs)
            "shap_values": shap_dict
        }

    except Exception as e:
        return {"error": str(e)}
    
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

@app.post("/predict-image")
async def predict_image(text: str = Form(""), image: UploadFile = File(None)):
    try:
        ocr_text = ""

        if image:
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))
            ocr_text = pytesseract.image_to_string(img)

        final_text = text + " " + ocr_text

        # Vectorize
        text_vector = vectorizer.transform([final_text])

        # Predict
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0][prediction]

        result = "FAKE" if prediction == 0 else "REAL"

        # ===== SHAP (correct) =====
        text_vector_dense = text_vector.toarray()
        shap_values = explainer(text_vector_dense)

        feature_names = vectorizer.get_feature_names_out()
        shap_dict = dict(zip(feature_names, shap_values.values[0]))

        # ===== LIME (for frontend) =====
        lime_exp = lime_explainer.explain_instance(
            final_text,
            predict_proba_lime,
            num_features=10
        )

        lime_dict = dict(lime_exp.as_list())

        # ===== Final explanation =====
        words = final_text.split()

        explanation = [
            {
                "word": w,
                "importance": float(lime_dict.get(w, 0))
            }
            for w in words
        ]

        return {
            "prediction": result,
            "confidence": round(float(probability) * 100, 2),
            "ocr_text": ocr_text,
            "final_text": final_text,

            # 👇 frontend uses this
            "explanation": explanation,

            # 👇 optional (for viva/debug)
            "shap_values": shap_dict
        }

    except Exception as e:
        return {"error": str(e)}

import torch
import numpy as np

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

MODEL_PATH = "bert_model"


def load_bert_components():
    """
    Loads the tokenizer, model, and label classes.
    Call this once from app.py, wrapped in @st.cache_resource,
    so it doesn't reload on every prediction / import.
    """

    label_classes = np.load("label_classes.npy", allow_pickle=True)

    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    return tokenizer, model, label_classes


def predict_student_emotion_bert(text, tokenizer, model, label_classes):
    """
    Predict student's emotion using the BERT model.

    Returns:
        dict containing:
            predicted_emotion
            confidence
    """

    try:

        if text is None:
            text = ""

        text = str(text).strip()

        if text == "":
            return {
                "predicted_emotion": "No Input",
                "confidence": 0.0
            }

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)

        prediction = torch.argmax(probabilities, dim=1).item()

        confidence = probabilities[0][prediction].item()

        emotion = str(label_classes[prediction])

        return {
            "predicted_emotion": emotion,
            "confidence": round(confidence, 4)
        }

    except Exception as e:

        return {
            "predicted_emotion": "Error",
            "confidence": 0.0,
            "error": str(e)
        }
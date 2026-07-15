import torch


# Emotion labels (Must match training labels)
EMOTION_LABELS = {
    0: "Bored",
    1: "Confident",
    2: "Confused",
    3: "Curious",
    4: "Frustrated"
}


def predict_student_emotion(model, text_pipeline, vocab, sentence, device):
    """
    Predict student's emotion from input text.

    Returns:
        dict containing:
            predicted_emotion
            confidence
            breakdown
    """

    try:

        model.eval()

        # -----------------------------
        # Validate input
        # -----------------------------
        if sentence is None:
            sentence = ""

        sentence = str(sentence).strip()

        if sentence == "":
            return {
                "predicted_emotion": "No Input",
                "confidence": 0.0,
                "breakdown": {}
            }

        # -----------------------------
        # Tokenization
        # -----------------------------
        tokens = text_pipeline(sentence)

        if not tokens:
            tokens = ["<unk>"]

        indexed = [
            vocab.get(token, vocab.get("<unk>", 1))
            for token in tokens
        ]

        if len(indexed) == 0:
            indexed = [1]

        text_tensor = torch.LongTensor(indexed).unsqueeze(0).to(device)

        lengths_tensor = torch.LongTensor([len(indexed)])

        # -----------------------------
        # Prediction
        # -----------------------------
        with torch.no_grad():

            outputs = model(text_tensor, lengths_tensor)

            probabilities = torch.softmax(outputs, dim=1).squeeze(0)

            predicted_index = int(torch.argmax(probabilities).item())

        # -----------------------------
        # Safety check
        # -----------------------------
        if predicted_index not in EMOTION_LABELS:
            return {
                "predicted_emotion": "Unknown",
                "confidence": 0.0,
                "breakdown": {}
            }

        prediction = EMOTION_LABELS[predicted_index]

        confidence = float(probabilities[predicted_index])

        breakdown = {}

        for i in range(len(EMOTION_LABELS)):
            breakdown[EMOTION_LABELS[i]] = round(
                float(probabilities[i]),
                4
            )

        return {
            "predicted_emotion": prediction,
            "confidence": round(confidence, 4),
            "breakdown": breakdown
        }

    except Exception as e:

        return {
            "predicted_emotion": "Error",
            "confidence": 0.0,
            "breakdown": {},
            "error": str(e)
        }
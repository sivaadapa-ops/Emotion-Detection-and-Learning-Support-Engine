import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# -----------------------------
# Emotion label mapping
# (must match predict.py's EMOTION_LABELS so BiLSTM and BERT agree)
# -----------------------------
EMOTION_MAP = {
    0: "Bored",
    1: "Confident",
    2: "Confused",
    3: "Curious",
    4: "Frustrated"
}

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset.csv")

# Keep required columns
df = df[["text", "label"]]

# Remove missing values
df.dropna(inplace=True)

# -----------------------------
# Map numeric labels to emotion names, then encode
# -----------------------------
df["label"] = df["label"].astype(int).map(EMOTION_MAP)

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# Save label classes (these will now be emotion names, e.g. "Confused")
np.save("label_classes.npy", label_encoder.classes_)

print("Label classes (index order):", list(label_encoder.classes_))
print("Total rows after cleaning:", len(df))
print("Class distribution:\n", df["label"].value_counts().sort_index())

# -----------------------------
# Split Dataset
# -----------------------------
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# -----------------------------
# Tokenization Function
# -----------------------------
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# -----------------------------
# Create HF Dataset
# -----------------------------
train_dataset = Dataset.from_dict({
    "text": list(train_texts),
    "labels": list(train_labels)
})

test_dataset = Dataset.from_dict({
    "text": list(test_texts),
    "labels": list(test_labels)
})

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# -----------------------------
# Load Model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_encoder.classes_)
)

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# -----------------------------
# Training Arguments
# -----------------------------
# CHANGES vs original:
# - num_train_epochs: 3 -> 10
#     With a dataset this size, 3 epochs gives too few optimization
#     steps for the randomly-initialized classification head to
#     converge, which is why confidence was stuck near random (~20%
#     for 5 classes).
# - added warmup_ratio=0.1
#     Ramps the learning rate up gradually over the first 10% of
#     training steps instead of hitting 2e-5 immediately, which
#     stabilizes early training.
# - added lr_scheduler_type="linear" (explicit, was implicit default)
# - load_best_model_at_end kept, now paired with more epochs so it
#   actually has multiple good checkpoints to choose from.
training_args = TrainingArguments(
    output_dir="bert_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    logging_dir="./logs",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Evaluate
# -----------------------------
results = trainer.evaluate()

print("\nEvaluation Results")
print(results)

# -----------------------------
# Save Model
# -----------------------------
trainer.save_model("bert_model")
tokenizer.save_pretrained("bert_model")

print("\nBERT model saved successfully!")
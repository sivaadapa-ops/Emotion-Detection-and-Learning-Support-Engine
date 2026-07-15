import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim

from model import BiLSTMEmotionClassifier
from vocab_builder import build_simple_vocab, basic_tokenizer


# -----------------------------
# Hyperparameters
# -----------------------------
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
OUTPUT_DIM = 5
EPOCHS = 35
LEARNING_RATE = 0.003
DROPOUT = 0.3


def train_model():

    print("=" * 60)
    print("Student Emotion Detection Training")
    print("=" * 60)

    csv_path = "dataset.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    # ---------------------------------
    # Build Vocabulary
    # ---------------------------------

    vocab = build_simple_vocab(csv_path)

    if len(vocab) <= 2:
        raise RuntimeError("Vocabulary could not be built.")

    print(f"Vocabulary Size : {len(vocab)}")

    # ---------------------------------
    # Read Dataset
    # ---------------------------------

    sentences = []
    labels = []

    with open(csv_path, "r", encoding="utf-8") as file:

        reader = csv.reader(file)

        next(reader, None)

        for row in reader:

            if len(row) < 2:
                continue

            text = row[0].strip()

            try:
                label = int(row[1])
            except ValueError:
                continue

            if label not in [0, 1, 2, 3, 4]:
                print(f"Skipping invalid label : {label}")
                continue

            tokens = basic_tokenizer(text)

            indexed = [
                vocab.get(token, vocab["<unk>"])
                for token in tokens
            ]

            if len(indexed) == 0:
                indexed = [vocab["<unk>"]]

            sentences.append(torch.LongTensor(indexed))
            labels.append(label)

    if len(sentences) == 0:
        raise RuntimeError("Dataset contains no valid training samples.")

    print(f"Training Samples : {len(sentences)}")

    # ---------------------------------
    # Model
    # ---------------------------------

    model = BiLSTMEmotionClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout_prob=DROPOUT
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=15,
        gamma=0.5
    )

    # ---------------------------------
    # Training
    # ---------------------------------

    print("\nTraining Started...\n")

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0

        for sentence, label in zip(sentences, labels):

            optimizer.zero_grad()

            inputs = sentence.unsqueeze(0)

            lengths = torch.LongTensor([sentence.size(0)])

            targets = torch.LongTensor([label])

            outputs = model(inputs, lengths)

            loss = criterion(outputs, targets)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        avg_loss = epoch_loss / len(sentences)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:02d}/{EPOCHS} "
                f" Loss : {avg_loss:.4f}"
            )

    # ---------------------------------
    # Save Model + Vocab + Config together
    # ---------------------------------

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "config": {
                "vocab_size": len(vocab),
                "embedding_dim": EMBEDDING_DIM,
                "hidden_dim": HIDDEN_DIM,
                "output_dim": OUTPUT_DIM,
                "dropout_prob": DROPOUT
            }
        },
        "bilstm_emotion_model.pt"
    )

    print("\nModel + vocab + config saved successfully to bilstm_emotion_model.pt")

    print("\nTraining Completed Successfully.")


if __name__ == "__main__":

    try:
        train_model()

    except Exception as e:
        print(f"\nTraining Failed : {e}")
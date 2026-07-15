import torch
import torch.nn as nn


class BiLSTMEmotionClassifier(nn.Module):
    """
    Bidirectional LSTM model for Student Emotion Detection.
    Multi-label version (returns logits for BCEWithLogitsLoss).
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        dropout_prob=0.3
    ):
        super(BiLSTMEmotionClassifier, self).__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than 0.")

        if output_dim <= 0:
            raise ValueError("output_dim must be greater than 0.")

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_prob
        )

        self.dropout = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(
            hidden_dim * 2,
            output_dim
        )

    def forward(self, text, text_lengths):
        """
        Forward pass.
        Returns raw logits for multi-label classification.
        """

        if text.size(0) == 0:
            raise ValueError("Empty batch received.")

        embedded = self.embedding(text)
        embedded = self.dropout(embedded)

        # Ensure lengths are on CPU (required by pack_padded_sequence)
        if text_lengths.device.type != "cpu":
            text_lengths = text_lengths.cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            text_lengths,
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        # Concatenate final forward and backward hidden states
        hidden = torch.cat(
            (
                hidden[-2],
                hidden[-1]
            ),
            dim=1
        )

        hidden = self.dropout(hidden)

        # Return raw output (logits). 
        # Do not apply sigmoid here. Sigmoid is applied inside BCEWithLogitsLoss.
        return self.fc(hidden)
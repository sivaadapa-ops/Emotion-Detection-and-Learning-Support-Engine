import re
import csv


def basic_tokenizer(text):
    """
    Clean and tokenize input text.

    - Converts text to lowercase
    - Preserves question marks
    - Removes unnecessary punctuation
    """

    if not isinstance(text, str):
        return []

    text = text.lower().strip()

    # Preserve question marks
    text = re.sub(r'(\?)', r' \1 ', text)

    # Remove punctuation except '?'
    text = re.sub(r"[.,!_():;\"'\-]", "", text)

    tokens = [token for token in text.split() if token]

    return tokens


def build_simple_vocab(csv_path):
    """
    Build vocabulary from dataset.

    Returns:
        dict : word -> index
    """

    vocab = {
        "<pad>": 0,
        "<unk>": 1
    }

    next_index = 2

    try:
        with open(csv_path, "r", encoding="utf-8") as file:

            reader = csv.reader(file)

            # Skip header
            next(reader, None)

            for row in reader:

                # Skip invalid rows
                if len(row) < 2:
                    continue

                text = row[0]

                if text is None:
                    continue

                text = text.strip()

                if text == "":
                    continue

                tokens = basic_tokenizer(text)

                for token in tokens:

                    if token not in vocab:
                        vocab[token] = next_index
                        next_index += 1

    except FileNotFoundError:
        print(f"[ERROR] Dataset not found: {csv_path}")

    except Exception as e:
        print(f"[ERROR] Failed to build vocabulary: {e}")

    return vocab


if __name__ == "__main__":

    dataset_path = "dataset.csv"

    vocab = build_simple_vocab(dataset_path)

    print(f"Vocabulary Size: {len(vocab)}")
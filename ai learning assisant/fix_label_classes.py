import os
import numpy as np

# Must match EMOTION_MAP / EMOTION_LABELS index order exactly
EMOTION_MAP = {
    0: "Bored",
    1: "Confident",
    2: "Confused",
    3: "Curious",
    4: "Frustrated"
}

LABEL_FILE = "label_classes.npy"


def main():
    if not os.path.exists(LABEL_FILE):
        print(f"'{LABEL_FILE}' not found in current folder: {os.getcwd()}")
        print("Make sure you run this script from the same folder as app.py.")
        return

    old_classes = np.load(LABEL_FILE, allow_pickle=True)
    print("BEFORE FIX - current label_classes.npy contents:")
    print(old_classes)
    print()

    correct_classes = np.array(
        [EMOTION_MAP[i] for i in range(len(EMOTION_MAP))],
        dtype=object
    )

    backup_path = LABEL_FILE + ".backup"
    np.save(backup_path, old_classes)
    print(f"Backed up old file to: {backup_path}")

    np.save(LABEL_FILE, correct_classes)

    print()
    print("AFTER FIX - new label_classes.npy contents:")
    print(np.load(LABEL_FILE, allow_pickle=True))
    print()
    print("Done. Restart your Streamlit app now:")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()
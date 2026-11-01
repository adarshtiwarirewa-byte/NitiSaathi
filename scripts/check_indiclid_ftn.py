
from pathlib import Path

import fasttext_new
fasttext_new.install_as_fasttext()

fasttext = fasttext_new


# IndicLID native-script model
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "models"
    / "indiclid"
    / "indiclid-ftn"
    / "model_baseline_roman.bin"
)


def contains_script(text: str, start: int, end: int) -> bool:
    """Check whether text contains characters from a Unicode script."""

    return any(start <= ord(char) <= end for char in text)


samples = [
    {
        "text": "यह योजना छोटे व्यवसायों को वित्तीय सहायता प्रदान करती है।",
        "script": "Devanagari",
        "start": 0x0900,
        "end": 0x097F,
        "expected": "__label__hin_Deva"
    },
    {
        "text": "தமிழ்நாடு அரசு புதிய நலத்திட்டத்தை அறிவித்துள்ளது.",
        "script": "Tamil",
        "start": 0x0B80,
        "end": 0x0BFF,
        "expected": "__label__tam_Tamil"
    }
]


print("Loading IndicLID FTN model...")

model = fasttext.load_model(str(MODEL_PATH))

print("Model loaded successfully!")


all_passed = True

for sample in samples:

    text = sample["text"]

    # Verify that the original Unicode characters survived.
    script_valid = contains_script(
        text,
        sample["start"],
        sample["end"]
    )

    print(f"\nScript: {sample['script']}")
    print("Unicode integrity:", "PASSED" if script_valid else "FAILED")

    if not script_valid:
        all_passed = False
        continue

    # Run actual language identification.
    labels, scores = model.predict(text, k=3)

# Print the raw output for debugging
    print("Raw labels:", repr(labels))
    print("Raw scores:", repr(scores))

    # Single-string input returns a tuple of label strings
    predicted_label = labels[0]
    confidence = float(scores[0])

    print("Top predictions:")

    for label, score in zip(labels, scores):
        print(f"  {label}: {float(score):.6f}")
    matches = predicted_label == sample["expected"]

    print("Expected:", sample["expected"])
    print("Predicted:", predicted_label)
    print("Score:", confidence)
    print("Prediction:", "PASSED" if matches else "NEEDS INVESTIGATION")

    if not matches:
        all_passed = False


print("\n--------------------------------")
print("FTN smoke test:", "PASSED" if all_passed else "NEEDS INVESTIGATION")
print("--------------------------------")

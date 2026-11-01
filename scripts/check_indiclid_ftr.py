
from pathlib import Path

import fasttext_new
fasttext_new.install_as_fasttext()

fasttext = fasttext_new


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "models"
    / "indiclid"
    / "indiclid-ftr"
    / "model_baseline_roman.bin"
)

samples = [
    {
        "text": "Mujhe is sarkari yojana ke liye aavedan kaise karna hai?",
        "expected": "__label__hin_Latn"
    },
    {
        "text": "Indha thittathukku yaar thagudhiyaanavanga?",
        "expected": "__label__tam_Latn"
    },
    {
        "text": (
            "Small businesses can apply for financial "
            "assistance under this government scheme."
        ),
        "expected": "__label__eng_Latn"
    }
]


print("Loading IndicLID FTR model...")

model = fasttext.load_model(str(MODEL_PATH))

print("Model loaded successfully!")

for sample in samples:

    text = sample["text"]

    labels, scores = model.predict(text, k=3)

    predicted = labels[0]
    score = float(scores[0])

    print("\nInput:", text)
    print("Expected:", sample["expected"])
    print("Predicted:", predicted)
    print("Raw model score:", score)

    print("Top predictions:")

    for label, probability in zip(labels, scores):
        print(f"  {label}: {float(probability):.6f}")

    print(
        "Result:",
        "PASSED" if predicted == sample["expected"]
        else "NEEDS INVESTIGATION"
    )

print("\nFTR inference test completed!")


from pathlib import Path
import re

import fasttext_new
fasttext_new.install_as_fasttext()

fasttext = fasttext_new
import torch
from transformers import PreTrainedTokenizerFast

from check_indiclid_bert_prediction import (
    LABELS,
    TOKENIZER_PATH,
    load_compatible_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ROOT = PROJECT_ROOT / "data/models/indiclid"

FTN_PATH = MODEL_ROOT / "indiclid-ftn/model_baseline_roman.bin"
FTR_PATH = MODEL_ROOT / "indiclid-ftr/model_baseline_roman.bin"

INPUT_THRESHOLD = 0.5
ROMAN_THRESHOLD = 0.6


SAMPLES = [
    "यह योजना छोटे व्यवसायों को वित्तीय सहायता प्रदान करती है।",
    "தமிழ்நாடு அரசு புதிய நலத்திட்டத்தை அறிவித்துள்ளது.",
    "Mujhe is sarkari yojana ke liye aavedan kaise karna hai?",
    "Indha thittathukku yaar thagudhiyaanavanga?",
    "Small businesses can apply for financial assistance under this government scheme.",
]


def calculate_roman_ratio(text: str) -> float:
    """Reproduce AI4Bharat's character-percentage routing logic."""

    special_pattern = re.compile(r'[@_!#$%^&*()<>?/\|}{~:]')

    special_chars = len(special_pattern.findall(text))
    spaces = len(re.findall(r'\s', text))
    newlines = len(re.findall(r'\n', text))

    total_chars = len(text) - special_chars - spaces - newlines

    if total_chars == 0:
        return 0.0

    latin_pattern = re.compile(r'[a-zA-Z0-9]')
    latin_chars = len(latin_pattern.findall(text))

    return latin_chars / total_chars


def predict_fasttext(model, text: str):
    """Return the top FastText label and its raw score."""

    labels, scores = model.predict(text, k=1)

    label = labels[0]

    if isinstance(label, (tuple, list)):
        label = label[0]

    if isinstance(label, bytes):
        label = label.decode("utf-8")

    if label.startswith("__label__"):
        label = label[9:]

    return label, float(scores[0])


def predict_with_bert(text, model, tokenizer):
    """Classify uncertain Roman-script text using BERT."""

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
        return_token_type_ids=True,
    )

    if "token_type_ids" not in encoded:
        raise RuntimeError("Tokenizer did not return token_type_ids.")

    with torch.inference_mode():
        outputs = model(
            input_ids=encoded["input_ids"],
            token_type_ids=encoded["token_type_ids"],
            attention_mask=encoded["attention_mask"],
        )

    index = outputs.logits.argmax(dim=1).item()

    return LABELS[index], outputs.logits[0, index].item()


def predict_language(text, ftn, ftr, bert, tokenizer):
    """Apply the complete IndicLID routing strategy."""

    roman_ratio = calculate_roman_ratio(text)

    # Route 1: Native-script detection.
    if roman_ratio <= INPUT_THRESHOLD:
        language, score = predict_fasttext(ftn, text)

        return {
            "language": language,
            "source": "IndicLID-FTN",
            "raw_score": score,
            "roman_ratio": roman_ratio,
        }

    # Route 2: First-stage Roman-script detection.
    language, score = predict_fasttext(ftr, text)

    if score > ROMAN_THRESHOLD:
        return {
            "language": language,
            "source": "IndicLID-FTR",
            "raw_score": score,
            "roman_ratio": roman_ratio,
        }

    # Route 3: BERT fallback for uncertain Roman-script text.
    language, logit = predict_with_bert(
        text,
        bert,
        tokenizer,
    )

    return {
        "language": language,
        "source": "IndicLID-BERT",
        "raw_score": logit,
        "roman_ratio": roman_ratio,
    }


def main():
    print("--- Loading IndicLID models ---")

    for path in (FTN_PATH, FTR_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    ftn = fasttext.load_model(str(FTN_PATH))
    print("FTN loaded.")

    ftr = fasttext.load_model(str(FTR_PATH))
    print("FTR loaded.")

    bert = load_compatible_model()
    print("BERT loaded.")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )
    print("Tokenizer loaded.")

    print("\n--- COMPLETE PIPELINE PREDICTIONS ---")

    for text in SAMPLES:
        result = predict_language(
            text,
            ftn,
            ftr,
            bert,
            tokenizer,
        )

        print("\nInput:", text)
        print("Detected language:", result["language"])
        print("Model used:", result["source"])
        print("Raw score:", result["raw_score"])
        print(
            "Latin-character ratio:",
            round(result["roman_ratio"], 3),
        )

    print("\nIndicLID pipeline smoke test completed.")


if __name__ == "__main__":
    main()

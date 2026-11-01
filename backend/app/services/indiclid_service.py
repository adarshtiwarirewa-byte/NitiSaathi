
from pathlib import Path
import re

import fasttext_new
fasttext_new.install_as_fasttext()

fasttext = fasttext_new
import torch

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_ROOT = PROJECT_ROOT / "data" / "models" / "indiclid"

FTN_PATH = (
    MODEL_ROOT
    / "indiclid-ftn"
    / "model_baseline_roman.bin"
)

FTR_PATH = (
    MODEL_ROOT
    / "indiclid-ftr"
    / "model_baseline_roman.bin"
)

BERT_PATH = (
    MODEL_ROOT
    / "indiclid-bert"
    / "basline_nn_simple.pt"
)

TOKENIZER_PATH = MODEL_ROOT / "tokenizer"


# --------------------------------------------------
# Official Roman-script BERT label mapping
# --------------------------------------------------

LABELS = {
    0: "asm_Latn",
    1: "ben_Latn",
    2: "brx_Latn",
    3: "guj_Latn",
    4: "hin_Latn",
    5: "kan_Latn",
    6: "kas_Latn",
    7: "kok_Latn",
    8: "mai_Latn",
    9: "mal_Latn",
    10: "mni_Latn",
    11: "mar_Latn",
    12: "nep_Latn",
    13: "ori_Latn",
    14: "pan_Latn",
    15: "san_Latn",
    16: "snd_Latn",
    17: "tam_Latn",
    18: "tel_Latn",
    19: "urd_Latn",
    20: "eng_Latn",
    21: "other",
}


# --------------------------------------------------
# Official routing thresholds
# --------------------------------------------------

INPUT_THRESHOLD = 0.5
ROMAN_THRESHOLD = 0.6

SPECIAL_PATTERN = re.compile(r'[@_!#$%^&*()<>?/\|}{~:]')
LATIN_PATTERN = re.compile(r'[a-zA-Z0-9]')


# --------------------------------------------------
# BERT compatibility loader
# --------------------------------------------------

def load_compatible_bert():
    """
    Load the official trained checkpoint into a fresh
    Transformers model while preserving its learned weights.

    The compatibility reconstruction is required because
    the original checkpoint is serialized using an older
    Transformers model structure.
    """

    print("Loading IndicLID-BERT checkpoint...")

    legacy_model = torch.load(
        BERT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    config_values = {
        key: value
        for key, value in vars(legacy_model.config).items()
        if not key.startswith("_")
    }

    config_values["num_labels"] = (
        legacy_model.classifier.out_features
    )

    config_values["architectures"] = [
        "BertForSequenceClassification"
    ]

    config_values["id2label"] = LABELS.copy()

    config_values["label2id"] = {
        label: index
        for index, label in LABELS.items()
    }

    config = BertConfig(**config_values)

    # Use standard attention implementation on CPU.
    config._attn_implementation = "eager"

    model = BertForSequenceClassification(config)

    state_dict = legacy_model.state_dict()

    # Verify and remove the legacy positional-index buffer.
    position_key = "bert.embeddings.position_ids"

    if position_key not in state_dict:
        raise KeyError(
            f"Missing expected checkpoint buffer: {position_key}"
        )

    checkpoint_positions = state_dict.pop(position_key)

    current_positions = model.bert.embeddings.position_ids

    if not torch.equal(
        checkpoint_positions.cpu(),
        current_positions.cpu(),
    ):
        raise RuntimeError(
            "Checkpoint position IDs do not match the new model."
        )

    # Transfer every remaining parameter strictly.
    load_result = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"BERT checkpoint mismatch: {load_result}"
        )

    model.eval()

    del state_dict
    del legacy_model

    print("IndicLID-BERT loaded successfully.")

    return model


# --------------------------------------------------
# IndicLID service
# --------------------------------------------------

class IndicLIDService:
    """
    Multilingual language identification using the official
    IndicLID-FTN, IndicLID-FTR and IndicLID-BERT models.
    """

    def __init__(self):

        # Verify model files before loading.
        required_paths = [
            FTN_PATH,
            FTR_PATH,
            BERT_PATH,
        ]

        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"IndicLID model not found: {path}"
                )

        if not TOKENIZER_PATH.exists():
            raise FileNotFoundError(TOKENIZER_PATH)

        print("Loading IndicLID-FTN...")
        self.ftn = fasttext.load_model(str(FTN_PATH))

        print("Loading IndicLID-FTR...")
        self.ftr = fasttext.load_model(str(FTR_PATH))

        self.bert = load_compatible_bert()

        print("Loading IndicLID tokenizer...")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            TOKENIZER_PATH,
            local_files_only=True,
        )

        self.device = torch.device("cpu")

        print("IndicLID service initialized successfully.")

    @staticmethod
    def calculate_latin_ratio(text: str) -> float:
        """
        Calculate the proportion of Latin letters and digits.

        Follows AI4Bharat's official script-routing logic.
        """

        input_length = len(text)

        special_chars = len(
            SPECIAL_PATTERN.findall(text)
        )

        spaces = len(re.findall(r'\s', text))
        newlines = len(re.findall(r'\n', text))

        total_chars = (
            input_length
            - special_chars
            - spaces
            - newlines
        )

        if total_chars <= 0:
            return 0.0

        latin_chars = len(
            LATIN_PATTERN.findall(text)
        )

        return latin_chars / total_chars

    @staticmethod
    def parse_label(label: str) -> tuple[str, str]:
        """Separate language and script from IndicLID's label."""

        if "_" not in label:
            return label, "Unknown"

        language_code, script = label.split("_", 1)

        return language_code, script

    @staticmethod
    def clean_fasttext_label(label) -> str:
        """Convert a fastText prediction into its language label."""

        if isinstance(label, bytes):
            label = label.decode("utf-8")

        if label.startswith("__label__"):
            label = label[9:]

        return label

    def predict_fasttext(self, model, text: str):
        """Return the top fastText label and its raw score."""

        labels, scores = model.predict(text, k=1)

        label = self.clean_fasttext_label(labels[0])

        raw_score = float(scores[0])

        return label, raw_score

    def predict_with_bert(self, text: str):
        """Classify uncertain Roman-script input using BERT."""

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_token_type_ids=True,
        )

        if "token_type_ids" not in encoded:
            raise RuntimeError(
                "Tokenizer did not return token_type_ids."
            )

        with torch.inference_mode():

            outputs = self.bert(
                input_ids=encoded["input_ids"].to(self.device),
                token_type_ids=encoded["token_type_ids"].to(
                    self.device
                ),
                attention_mask=encoded["attention_mask"].to(
                    self.device
                ),
            )

        predicted_index = outputs.logits.argmax(dim=1).item()

        language_label = LABELS[predicted_index]

        raw_logit = outputs.logits[
            0, predicted_index
        ].item()

        return language_label, raw_logit

    def detect(self, text: str) -> dict:
        """
        Detect the language and script of a text passage.

        Native script -> FTN
        Roman script -> FTR
        Uncertain Roman script -> BERT
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Language detection requires non-empty text."
            )

        text = text.strip()

        latin_ratio = self.calculate_latin_ratio(text)

        # Stage 1: Native-script detection.
        if latin_ratio <= INPUT_THRESHOLD:

            language_label, raw_score = self.predict_fasttext(
                self.ftn,
                text,
            )

            detector = "IndicLID-FTN"

        else:

            # Stage 2: Roman-script fastText detection.
            language_label, raw_score = self.predict_fasttext(
                self.ftr,
                text,
            )

            if raw_score > ROMAN_THRESHOLD:

                detector = "IndicLID-FTR"

            else:

                # Stage 3: BERT fallback.
                language_label, raw_score = (
                    self.predict_with_bert(text)
                )

                detector = "IndicLID-BERT"

        language_code, script = self.parse_label(
            language_label
        )

        return {
            "language_label": language_label,
            "language_code": language_code,
            "script": script,
            "detection_model": detector,
            "detection_raw_score": float(raw_score),
            "latin_ratio": float(latin_ratio),
        }

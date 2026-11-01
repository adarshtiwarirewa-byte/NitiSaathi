
from pathlib import Path
import torch

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "data/models/indiclid/indiclid-bert"
    / "basline_nn_simple.pt"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "data/models/indiclid/tokenizer"
)


# Official IndicLID-BERT Roman-script output labels.
# Verified against AI4Bharat's official inference implementation.
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


SAMPLES = [
    "Mujhe is sarkari yojana ke liye aavedan kaise karna hai?",
    "Indha thittathukku yaar thagudhiyaanavanga?",
    "Small businesses can apply for financial assistance under this government scheme.",
]


def load_compatible_model():
    """Load the original checkpoint into a fresh compatible BERT model."""

    print("Loading original IndicLID-BERT checkpoint...")

    legacy_model = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    # Copy public configuration values from the original model.
    config_values = {
        key: value
        for key, value in vars(legacy_model.config).items()
        if not key.startswith("_")
    }

    # Preserve the original classifier dimensions and labels.
    config_values["num_labels"] = legacy_model.classifier.out_features
    config_values["architectures"] = [
        "BertForSequenceClassification"
    ]
    config_values["id2label"] = LABELS.copy()
    config_values["label2id"] = {
        label: index
        for index, label in LABELS.items()
    }

    # Create a fresh configuration for the installed Transformers version.
    config = BertConfig(**config_values)
    config._attn_implementation = "eager"

    # Create a new model with the same architecture.
    model = BertForSequenceClassification(config)

    # Retrieve all original checkpoint parameters and buffers.
    state_dict = legacy_model.state_dict()

    # Handle the old checkpoint's positional-index buffer.
    position_key = "bert.embeddings.position_ids"

    if position_key not in state_dict:
        raise KeyError(
            f"Expected checkpoint buffer missing: {position_key}"
        )

    checkpoint_positions = state_dict.pop(position_key)
    current_positions = model.bert.embeddings.position_ids

    # Verify that both models use exactly the same position indices.
    if not torch.equal(
        checkpoint_positions.cpu(),
        current_positions.cpu(),
    ):
        raise RuntimeError(
            "Position IDs differ between original and new model."
        )

    print("Position IDs verified successfully.")

    # Strict loading ensures no other parameter mismatch is ignored.
    load_result = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: {load_result}"
        )

    print("Original learned weights transferred successfully.")

    model.eval()

    # Release the original model and temporary state dictionary.
    del state_dict
    del legacy_model

    print("Model class:", type(model).__name__)
    print("Classifier output size:", model.classifier.out_features)
    print("Attention backend:", model.config._attn_implementation)

    return model


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(TOKENIZER_PATH)

    # Step 1: Load the compatible model.
    model = load_compatible_model()

    # Step 2: Load the locally saved tokenizer.
    print("\nLoading IndicBERT tokenizer...")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )

    # Step 3: Tokenize the test sentences.
    encoded = tokenizer(
        SAMPLES,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_token_type_ids=True,
    )

    if "token_type_ids" not in encoded:
        raise RuntimeError(
            "Tokenizer did not produce token_type_ids."
        )

    print("\nTokenizer loaded successfully.")

    for key, value in encoded.items():
        print(f"{key}: {tuple(value.shape)}")

    # Step 4: Run actual BERT inference.
    print("\nRunning BERT predictions...")

    with torch.inference_mode():
        outputs = model(
            input_ids=encoded["input_ids"],
            token_type_ids=encoded["token_type_ids"],
            attention_mask=encoded["attention_mask"],
        )

    # Step 5: Interpret the model outputs.
    print("\n--- PREDICTIONS ---")
    print("Logits shape:", tuple(outputs.logits.shape))

    predicted_indices = torch.argmax(
        outputs.logits,
        dim=1,
    )

    for sample, index, logits in zip(
        SAMPLES,
        predicted_indices,
        outputs.logits,
    ):
        label_index = index.item()
        raw_logit = logits[label_index].item()

        print("\nText:", sample)
        print("Predicted label:", LABELS[label_index])
        print("Predicted index:", label_index)
        print("Winning raw logit:", raw_logit)

    # Final validation.
    expected_shape = (len(SAMPLES), len(LABELS))

    if tuple(outputs.logits.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected logits shape: {tuple(outputs.logits.shape)}"
        )

    print("\nBERT prediction smoke test completed successfully.")


if __name__ == "__main__":
    main()

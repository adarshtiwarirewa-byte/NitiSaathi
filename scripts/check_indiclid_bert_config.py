from pathlib import Path
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "data/models/indiclid/indiclid-bert"
    / "basline_nn_simple.pt"
)

TOKENIZER_PATH = (
    PROJECT_ROOT / "data/models/indiclid/tokenizer"
)


def main():
    print("Loading BERT checkpoint...")

    model = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    config = model.config

    print("\n--- MODEL CONFIGURATION ---")
    print("Model type:", config.model_type)
    print("Architecture:", config.architectures)
    print("Number of labels:", config.num_labels)
    print("Hidden size:", config.hidden_size)
    print("Transformer layers:", config.num_hidden_layers)
    print("Maximum positions:", config.max_position_embeddings)

    print("\n--- MODEL LABEL MAPPING ---")

    for index in range(config.num_labels):
        label = config.id2label.get(
            index,
            config.id2label.get(str(index), "MISSING")
        )
        print(f"{index}: {label}")

    print("\n--- TOKENIZER CONFIGURATION ---")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )

    print("Tokenizer class:", type(tokenizer).__name__)
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Maximum length:", tokenizer.model_max_length)
    print("Model input names:", tokenizer.model_input_names)

    sample = (
        "Mujhe is sarkari yojana ke liye "
        "aavedan kaise karna hai?"
    )

    encoded = tokenizer(
        sample,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_token_type_ids=True,
    )

    print("\n--- TOKENIZER OUTPUT ---")

    for key, value in encoded.items():
        print(f"{key}: {tuple(value.shape)}")

    print("\nConfiguration inspection completed.")

    del model


if __name__ == "__main__":
    main()
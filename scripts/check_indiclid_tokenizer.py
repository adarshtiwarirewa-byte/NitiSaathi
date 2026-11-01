
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"

TOKENIZER_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "indiclid"
    / "tokenizer"
)


samples = [
    "यह योजना छोटे व्यवसायों को वित्तीय सहायता प्रदान करती है।",
    "தமிழ்நாடு அரசு புதிய நலத்திட்டத்தை அறிவித்துள்ளது.",
    "Mujhe is sarkari yojana ke liye aavedan karna hai.",
    "Small businesses can apply for financial assistance."
]


print("Loading official IndicBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully!")
print("Tokenizer class:", type(tokenizer).__name__)
print("Vocabulary size:", tokenizer.vocab_size)


# Tokenize all four language examples
encoded = tokenizer(
    samples,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
    return_token_type_ids=True
)

print("\nTokenization results:")

for index, sample in enumerate(samples):

    print(f"\nExample {index + 1}:")
    print("Original:", sample)
    print("Token count:", int(encoded["attention_mask"][index].sum()))


# Verify inputs required by official IndicLID inference
required_keys = {
    "input_ids",
    "attention_mask",
    "token_type_ids"
}

missing_keys = required_keys - set(encoded.keys())

if missing_keys:
    raise RuntimeError(
        f"Required tokenizer outputs missing: {missing_keys}"
    )


print("\nAll required tokenizer outputs are present.")


# Save tokenizer locally
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

tokenizer.save_pretrained(TOKENIZER_DIR)

print("\nTokenizer saved at:", TOKENIZER_DIR)


# Verify that the saved tokenizer works without internet access
local_tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_DIR,
    local_files_only=True
)

local_encoded = local_tokenizer(
    samples,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)

assert local_encoded["input_ids"].equal(encoded["input_ids"])

print("Local tokenizer verification: PASSED")
print("\nIndicBERT tokenizer test completed!")

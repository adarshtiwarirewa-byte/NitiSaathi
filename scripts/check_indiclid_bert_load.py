from pathlib import Path
import gc
import time
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "indiclid"
    / "indiclid-bert"
    / "basline_nn_simple.pt"
)


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"BERT checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("PyTorch version:", torch.__version__)
    print("Device: CPU")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("\nLoading official IndicLID-BERT checkpoint...")

    start_time = time.perf_counter()

    # This is a trusted official checkpoint whose SHA256
    # checksum you have already verified.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    elapsed = time.perf_counter() - start_time

    print("\nCHECKPOINT LOADED SUCCESSFULLY")
    print("Object type:", type(checkpoint))
    print(f"Load time: {elapsed:.2f} seconds")

    if isinstance(checkpoint, dict):
        print("Dictionary keys:", list(checkpoint.keys())[:20])

    del checkpoint
    gc.collect()

    print("\nCheckpoint released from memory.")


if __name__ == "__main__":
    main()
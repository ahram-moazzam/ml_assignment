"""Stage 2 - normalize pixels to [0, 1], split train/val, save to data/processed/."""
import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "processed")


def normalize(x):
    """Teammate: per-image min-max scaling."""
    x = x.astype("float32")
    mn = x.min(axis=(1, 2), keepdims=True)
    mx = x.max(axis=(1, 2), keepdims=True)
    return (x - mn) / (mx - mn + 1e-8)

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    x_train = np.load(os.path.join(RAW_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(RAW_DIR, "y_train.npy"))
    x_test = np.load(os.path.join(RAW_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(RAW_DIR, "y_test.npy"))

    x_train, x_test = normalize(x_train), normalize(x_test)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=y_train,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, arr in [("x_train", x_train), ("y_train", y_train),
                      ("x_val", x_val), ("y_val", y_val),
                      ("x_test", x_test), ("y_test", y_test)]:
        np.save(os.path.join(OUT_DIR, f"{name}.npy"), arr)

    print(f"Saved processed data to {OUT_DIR}: "
          f"train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")


if __name__ == "__main__":
    main()

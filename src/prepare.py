
import os
import numpy as np
from tensorflow import keras

OUT_DIR = os.path.join("data", "raw")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    np.save(os.path.join(OUT_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)

    print(f"Saved raw data to {OUT_DIR}: train={x_train.shape}, test={x_test.shape}")


if __name__ == "__main__":
    main()

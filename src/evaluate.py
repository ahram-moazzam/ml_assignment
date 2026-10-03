import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras

DATA_DIR = os.path.join("data", "processed")
MODEL_PATH = os.path.join("models", "model.h5")
REPORT_DIR = "reports"

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def main():
    x_test = np.load(os.path.join(DATA_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    model = keras.models.load_model(MODEL_PATH)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    with open("metrics.json", "w") as f:
        json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    os.makedirs(REPORT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=120)

    print(f"Test loss={loss:.4f}  Test accuracy={acc:.4f}")


if __name__ == "__main__":
    main()

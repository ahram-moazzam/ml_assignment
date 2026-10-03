
import csv
import os
import numpy as np
import tensorflow as tf
import yaml
from tensorflow import keras

DATA_DIR = os.path.join("data", "processed")
MODEL_DIR = "models"


def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["train"]

    tf.keras.utils.set_random_seed(p["seed"])

    x_train = np.load(os.path.join(DATA_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    x_val = np.load(os.path.join(DATA_DIR, "x_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(p["dense_units"], activation="relu"),
        keras.layers.Dropout(p["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=p["epochs"],
        batch_size=p["batch_size"],
        verbose=2,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "model.h5"))

    hist = history.history
    with open(os.path.join(MODEL_DIR, "history.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + list(hist.keys()))
        for i in range(len(hist["loss"])):
            writer.writerow([i + 1] + [hist[k][i] for k in hist])

    print(f"Saved model and history to {MODEL_DIR}/")


if __name__ == "__main__":
    main()

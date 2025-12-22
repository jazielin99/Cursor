#!/usr/bin/env python3
"""
CNN feature extractor for a single image (MobileNetV2, 1,280 dims).

Writes a one-row CSV with columns cnn_0..cnn_1279 plus `path`.
Used by Prediction_New/predict_new.R.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

IMG_SIZE = 224


def create_feature_extractor():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False
    return base_model


def load_image(img_path: str) -> np.ndarray:
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = image.img_to_array(img)
    arr = preprocess_input(arr)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CNN (MobileNetV2) features for one image.")
    parser.add_argument("--image", required=True, help="Image path")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    args = parser.parse_args()

    model = create_feature_extractor()
    arr = load_image(args.image)
    feats = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]

    cols = [f"cnn_{i}" for i in range(feats.shape[0])]
    df = pd.DataFrame([feats], columns=cols)
    df["path"] = args.image
    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    # Reduce TF logging noise
    tf.get_logger().setLevel("ERROR")
    main()


import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

DEFAULT_LABELS = ["CNV", "DME", "DRUSEN", "NORMAL"]

def load_model(model_path: str):
    # Works for .keras/.h5 files or SavedModel dirs
    model = tf.keras.models.load_model(model_path, compile=False)
    # Try to infer input size/channels from the first input tensor
    input_shape = model.inputs[0].shape.as_list()
    # input_shape: [None, H, W, C]
    h = int(input_shape[1]) if input_shape[1] is not None else 256
    w = int(input_shape[2]) if input_shape[2] is not None else 256
    c = int(input_shape[3]) if input_shape[3] is not None else 1
    return model, (h, w, c)

def preprocess_image(path: str, target_hw_c, normalize=True):
    h, w, c = target_hw_c
    # Read as grayscale, then adapt to channels needed
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.resize(img_gray, (w, h), interpolation=cv2.INTER_AREA).astype("float32")
    if normalize:
        img = img / 255.0
    if c == 1:
        img = np.expand_dims(img, axis=-1)  # (H,W,1)
    elif c == 3:
        img = np.repeat(img[..., None], 3, axis=-1)  # gray->RGB
    else:
        raise ValueError(f"Unsupported channel count in model input: {c}")
    return img

def predict_paths(model, target_hw_c, paths, labels):
    xs, names = [], []
    for p in paths:
        xs.append(preprocess_image(p, target_hw_c))
        names.append(os.path.basename(p))
    x = np.stack(xs, axis=0)  # (N,H,W,C)
    probs = model.predict(x, verbose=0)
    preds = np.argmax(probs, axis=1).tolist()
    results = []
    for name, pr, idx in zip(names, probs, preds):
        results.append({
            "image": name,
            "pred_label": labels[idx],
            "pred_scores": {lbl: float(pr[i]) for i, lbl in enumerate(labels)}
        })
    return results

def main():
    ap = argparse.ArgumentParser(description="Minimal OCT classification demo")
    ap.add_argument("--model", required=True, help="Path to saved Keras model (.keras/.h5 or SavedModel dir)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="Path to a single image")
    g.add_argument("--dir", help="Directory of images (jpg/png)")
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS, help="Class labels in model order")
    ap.add_argument("--output", default="demo_predictions.json", help="Where to save JSON results")
    args = ap.parse_args()

    model, target_hw_c = load_model(args.model)

    if args.image:
        paths = [args.image]
    else:
        paths = [str(p) for p in Path(args.dir).glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
        paths.sort()

    if not paths:
        raise SystemExit("No images found to run inference on.")

    results = predict_paths(model, target_hw_c, paths, args.labels)

    # Pretty print to console
    for r in results:
        best = r["pred_label"]
        print(f"{r['image']}: {best} | " +
              ", ".join([f"{k}={r['pred_scores'][k]:.3f}" for k in args.labels]))

    # Save JSON
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved predictions → {args.output}")

if __name__ == "__main__":
    main()

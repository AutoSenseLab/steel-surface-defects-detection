import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

IMG_SIZE   = (224, 224)
MODEL_PATH = "models/best_model.h5"

CLASS_NAMES = ["Crazing", "Inclusion", "Patches", "Pitted_surface", "Rolled-in_scale", "Scratches"]


def load_model(model_path=MODEL_PATH):
    return tf.keras.models.load_model(model_path)


def preprocess_image(image_input):
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError("Expected a file path or PIL Image")

    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32), img


def predict(image_input, model, class_names=CLASS_NAMES):
    img_array, _ = preprocess_image(image_input)
    probs        = model.predict(img_array, verbose=0)[0]
    pred_idx     = np.argmax(probs)

    return {
        "class":      class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probs":  {cls: float(p) for cls, p in zip(class_names, probs)}
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    model  = load_model()
    result = predict(sys.argv[1], model)

    print(f"\nClass:      {result['class']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"\nAll probabilities:")
    for cls, prob in sorted(result['all_probs'].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"  {cls:20s}: {prob*100:5.1f}% {bar}")

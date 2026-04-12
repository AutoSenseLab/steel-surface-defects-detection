import numpy as np
import tensorflow as tf
import cv2


def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """
    img_array: (1, H, W, 3) normalized 0-1
    Returns heatmap: (H', W') values 0-1
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer(model)

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_class = tf.argmax(predictions[0])
        loss = predictions[:, pred_class]

    grads        = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_heatmap(original_img, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    original_img: (H, W, 3) uint8
    Returns (overlaid, heatmap_colored)
    """
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_color   = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_color   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlaid = cv2.addWeighted(
        original_img.astype(np.uint8), 1 - alpha,
        heatmap_color, alpha, 0
    )
    return overlaid, heatmap_color

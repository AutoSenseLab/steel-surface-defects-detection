import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2 , EfficientNetV2B0 , ResNet50
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 30
DATA_DIR   = "data/processed"
MODEL_PATH = "models/best_model.h5"
PLOT_PATH  = "models/training_history.png"


def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen


def build_model(num_classes):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'],     label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'],     label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Plot saved → {PLOT_PATH}")


def train():
    os.makedirs("models", exist_ok=True)

    print("Loading data...")
    train_gen, val_gen = build_generators()
    num_classes = len(train_gen.class_indices)
    print(f"Classes ({num_classes}): {train_gen.class_indices}\n")

    print("Building model...")
    model = build_model(num_classes)
    model.summary()

    initial_learning_rate = 1e-3
    opt = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: float(lr * 0.9048))
        
    ]
    
    model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',  # 'binary_crossentropy' for multi-label classification
    metrics=['accuracy']
)

    model.summary()

    # Unfreeze more layers for fine-tuning
    model.trainable = True
    fine_tune_at = 50
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False

    # Adjust the learning rate for fine-tuning
    opt_fine_tune = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=opt_fine_tune,
        loss='categorical_crossentropy',  # 'binary_crossentropy' for multi-label classification
        metrics=['accuracy']
    )

    print("\nStarting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    plot_history(history)
    print(f"\nModel saved → {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()

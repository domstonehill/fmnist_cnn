import tensorflow as tf
import tensorflow.keras.layers as layers
import visualkeras
from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
from configs import *
import matplotlib.pyplot as plt

# Autotune
AUTOTUNE = tf.data.AUTOTUNE

# Reshape and Rescaling Layers
reshape_and_rescale = tf.keras.Sequential([
    layers.Reshape((28, 28, 1)),
    # layers.Resizing(28, 28),
    layers.Rescaling(1. / 255)
])


# Augmentation
augmentation = tf.keras.Sequential([
    layers.RandomBrightness(factor=0.5, value_range=(0.0, 1.0)),
    layers.RandomContrast(factor=0.5),
    layers.RandomFlip(mode='horizontal')
])


# Data Loading Pipeline
def load_dataset(filename, train=False):
    # Read in CSV File
    features = pd.read_csv(DATA_DIR + filename)

    # Pop out the label column to use as labels
    labels = features.pop('label')

    # features = features.to_numpy()

    # Build Dataset from tensor slices and pass in the features and labels as numpy arrays
    # NOTE: not one-hot encoding labels, so will need to use sparse-categorical crossentropy
    data = tf.data.Dataset.from_tensor_slices((features.to_numpy(), labels.to_numpy()))

    # Batch
    data = data.batch(64)

    # Shuffle
    data = data.shuffle(buffer_size=1000)

    # Reshape and Rescale
    data = data.map(
        lambda x, y: (reshape_and_rescale(x), y),
        num_parallel_calls=AUTOTUNE
    )

    if train:
        data = data.map(
            lambda x, y: (augmentation(x), y),
            num_parallel_calls=AUTOTUNE
        )

    # Prefetch
    data = data.prefetch(buffer_size=AUTOTUNE)

    return data


def build_cnn_model():
    model = tf.keras.Sequential([
        layers.Conv2D(10, kernel_size=(5, 5), input_shape=(28, 28, 1), padding='same', activation='relu'),
        layers.MaxPool2D(),
        # layers.Dropout(0.25),
        layers.Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu'),
        layers.MaxPool2D(),
        # layers.Dropout(0.25),
        layers.Conv2D(28, kernel_size=(5, 5), padding='same', activation='relu'),
        layers.MaxPool2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy()
        ]
    )

    return model


def build_parallel_cnn():
    inputs = layers.Input(shape=(28, 28, 1))

    # 3x3 convolutions
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPool2D(padding='same')(x)
    x = layers.Dropout(0.5)(x)

    # 5x5 Convolutions
    y = layers.Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu')(inputs)
    y = layers.MaxPool2D(padding='same')(y)
    y = layers.Dropout(0.5)(y)

    # Concat
    cat = layers.Concatenate(axis=1)([x, y])

    # Flatten
    d = layers.Flatten()(cat)

    # Dense Network
    d = layers.Dense(128, activation='relu')(d)
    d = layers.Dropout(0.5)(d)

    # Output
    output = layers.Dense(10, activation='softmax')(d)

    # Model
    model = tf.keras.Model(inputs, output)

    # Compile
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy()
        ]
    )

    return model


def avg_pool_cnn():

    # Input Layer
    inputs = layers.Input(shape=(28, 28, 1))

    # Convolutions
    x = layers.Conv2D(10, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = layers.AveragePooling2D()(x)
    # x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.AveragePooling2D()(x)
    # x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(28, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Flatten and Dense Layers
    flat = layers.Flatten()(x)
    d = layers.Dense(128, activation='relu')(flat)
    d = layers.Dense(64, activation='relu')(d)

    # Output Layers
    output = layers.Dense(10, activation='softmax')(d)

    # Model Definition
    model = tf.keras.Model(inputs, output)

    # Model Compilation
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy()
        ]
    )

    return model


if __name__ == '__main__':
    train_data = load_dataset('fashion-mnist_train.csv', False)
    test_data = load_dataset('fashion-mnist_test.csv')

    # model = build_cnn_model()
    # model = build_parallel_cnn()
    model = avg_pool_cnn()

    print(model.summary())
    plot_model(model)
    visualkeras.layered_view(model, to_file='model_layered.png')

    history = model.fit(
        train_data,
        epochs=25,
        verbose=1,
        validation_data=test_data
    )

    # Use pandas to save history to csv
    hist = pd.DataFrame(history.history)
    hist.to_csv('history.csv')

    # Plot History Graph
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')

    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(['Train', 'Validation'])

    ax2.plot(history.history['sparse_categorical_accuracy'])
    ax2.plot(history.history['val_sparse_categorical_accuracy'])
    ax2.set_title('Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])

    plt.show()

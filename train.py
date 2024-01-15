import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.applications.resnet50 as resnet50

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


SEED = 29
SELECTED_CATEGORIES = ["Octopus", "Penguin", "Otter", "Starfish"]


def get_category_counts(cleaned_data_dir: str):
    total_images = 0
    cat_counts = {}

    for dirpath, dirnames, filenames in os.walk(cleaned_data_dir):
        if len(dirnames) == 0:
            total_images += len(filenames)
            cat_counts[os.path.basename(dirpath)] = len(filenames)

    print(f"{total_images=}")
    print(f"Total categories: {len(cat_counts.keys())}")
    print(f"Images per category: {cat_counts}")

    return cat_counts


def canonicalize_category(cat_name):
    return cat_name.lower().replace(" ", "_")


def prepare_df(cat_counts: dict, cleaned_data_dir: str):
    columns = ["filename", "class"]

    full_data = []

    for cat in cat_counts.keys():
        if cat not in SELECTED_CATEGORIES:
            continue
        canon_cat = canonicalize_category(cat)
        full_ds = tf.data.Dataset.list_files(f"{cleaned_data_dir}/seacreatures/{cat}/*")

        full_data.extend([(f.numpy().decode("utf-8"), canon_cat) for f in full_ds])

    full_df = pd.DataFrame(full_data, columns=columns).sample(frac=1)
    full_df.describe()

    full_train_df, test_df = train_test_split(
        full_df, test_size=0.2, shuffle=True, random_state=SEED
    )
    train_df, val_df = train_test_split(
        full_train_df, test_size=0.25, shuffle=True, random_state=SEED
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def prepare_data_generators(train_df: pd.DataFrame, val_df: pd.DataFrame):
    cwd = os.getcwd()

    train_gen_resnet = ImageDataGenerator(preprocessing_function=resnet50.preprocess_input)

    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_ds_resnet = train_gen_resnet.flow_from_dataframe(
        train_df,
        directory=cwd,
        target_size=(150, 150),
        batch_size=32,
        shuffle=True,
        seed=SEED,
        class_mode="categorical",
    )

    val_ds_resnet = val_gen.flow_from_dataframe(
        val_df,
        directory=cwd,
        target_size=(150, 150),
        batch_size=32,
        shuffle=True,
        seed=SEED,
        class_mode="categorical",
    )

    return train_ds_resnet, val_ds_resnet


def plot_history(history, model_name: str):
    h = history.history
    acc = h["accuracy"]
    val_acc = h["val_accuracy"]
    loss = h["loss"]
    val_loss = h["val_loss"]

    epochs = range(1, len(acc) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "bo", label="train acc")
    plt.plot(epochs, val_acc, "b", label="val acc")
    plt.title(f"Train/Val Accuracy ({model_name})")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo", label="train loss")
    plt.plot(epochs, val_loss, "b", label="val loss")
    plt.title(f"Train/Val Loss ({model_name})")
    plt.legend()

    plt.show()


def plot_val_acc(scores, epochs=15):
    for label, hist in scores.items():
        plt.plot(hist.history["val_accuracy"], label=f"val_{label}")

    plt.xticks(np.arange(epochs))
    plt.legend()


def make_model(
    input_shape,
    num_classes,
    dense_units=32,
    with_dropout=False,
    dropout=0.5,
    learning_rate=0.001,
    with_batch_normalization=False,
    with_second_dense_layer=False,
    layers_to_unfreeze=0,
):
    base_model = keras.applications.resnet50.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    for layer in base_model.layers:
        layer.trainable = False

    if layers_to_unfreeze > 0:
        # freeze up to the last 8 layers, to allow base model to be tuned
        unfreeze_after_layer = 50 - layers_to_unfreeze
        for layer in base_model.layers[unfreeze_after_layer:]:
            layer.trainable = True

    model = keras.models.Sequential()
    model.add(base_model)

    model.add(keras.layers.GlobalAveragePooling2D())

    if with_batch_normalization:
        model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(dense_units, activation="relu"))

    if with_dropout:
        model.add(keras.layers.Dropout(dropout))

    if with_second_dense_layer:
        model.add(keras.layers.Dense(dense_units // 2, activation="relu"))

        if with_dropout:
            model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def main(cleaned_data_dir: str):
    # Data preparation
    cat_counts = get_category_counts(cleaned_data_dir)
    train_df, val_df, _ = prepare_df(cat_counts, cleaned_data_dir)
    train_ds, val_ds = prepare_data_generators(train_df, val_df)
    num_classes = len(train_ds.class_indices)

    # Tuning parameters
    learning_rate = 0.001
    with_dropout = True
    dropout = 0.6
    epochs = 10
    dense_units = 256
    with_batch_norm = True
    with_2nd_dense_layer = True

    # Create and train model
    model = make_model(
        (150, 150, 3),
        num_classes,
        with_dropout=with_dropout,
        dropout=dropout,
        dense_units=dense_units,
        learning_rate=learning_rate,
        with_batch_normalization=with_batch_norm,
        with_second_dense_layer=with_2nd_dense_layer,
    )
    model.summary()
    model.fit(train_ds, epochs=epochs, validation_data=val_ds)

    # Convert model to tflite and save
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as fp:
        fp.write(tflite_model)


if __name__ == "__main__":
    main("cleaned_padded_data")

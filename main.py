#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import random
import sys


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Lambda,
)
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img

SEED = 42
INPUT_SIZE = (224, 224, 3)
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
ALPHA = 0.2
OUTPUT_DIM = 64
BASE_RELATIVE_IMAGE_PATH = "/images/"
BATCH_SIZE = 16


def generate_images(files, batch_size):
    L = len(files[0].values)
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            anchor = list()
            positive = list()
            negative = list()
            y = list()

            for i in range(batch_start, limit):
                anchor.append(
                    preprocess_input(
                        img_to_array(
                            load_img(
                                f"{BASE_RELATIVE_IMAGE_PATH + files[0].values[i]}.jpg",
                                target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                            )
                        )
                    )
                )

                positive.append(
                    preprocess_input(
                        img_to_array(
                            load_img(
                                f"{BASE_RELATIVE_IMAGE_PATH + files[1].values[i]}.jpg",
                                target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                            )
                        )
                    )
                )

                negative.append(
                    preprocess_input(
                        img_to_array(
                            load_img(
                                f"{BASE_RELATIVE_IMAGE_PATH + files[2].values[i]}.jpg",
                                target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                            )
                        )
                    )
                )

            yield (
                [np.array(anchor), np.array(positive), np.array(negative)],
                np.array([0 for i in range(len(anchor))]),
            )

            batch_start += batch_size
            batch_end += batch_size


def triplet_loss_batch_all(y_true, y_pred):
    # Source
    # https://omoindrot.github.io/triplet-loss
    anchor_embed = y_pred[:, :OUTPUT_DIM]
    positive_embed = y_pred[:, OUTPUT_DIM : 2 * OUTPUT_DIM]
    negative_embed = y_pred[:, 2 * OUTPUT_DIM :]
    d_pos = tf.reduce_sum(K.abs(anchor_embed - positive_embed), 1)
    d_neg = tf.reduce_sum(K.abs(anchor_embed - negative_embed), 1)
    loss = tf.maximum(0.0, ALPHA + d_pos - d_neg)

    valid_triplets = tf.cast(tf.greater(loss, 1e-16), dtype=tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)

    triplet_loss = tf.reduce_sum(loss) / (num_positive_triplets + 1e-16)
    return triplet_loss


def create_custom_cnn(input_shape):
    input = Input(shape=input_shape)
    model = VGG16(include_top=False, weights="imagenet", input_tensor=input)
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(OUTPUT_DIM, activation=None)(x)
    embedding = Lambda(lambda f: tf.linalg.normalize(f, axis=1, ord=1)[0])(x)
    model = Model(input, embedding)
    model.summary()
    return model


def create_triplet_network(base_cnn, input_shape):
    anchor = Input(input_shape)
    positive_example = Input(input_shape)
    negative_example = Input(input_shape)

    anchor_model = base_cnn(anchor)
    pos_model = base_cnn(positive_example)
    neg_model = base_cnn(negative_example)
    embeddings = Concatenate(axis=1)([anchor_model, pos_model, neg_model])
    siamese_triplet = Model(
        inputs=[anchor, positive_example, negative_example], outputs=embeddings,
    )

    return siamese_triplet


def main():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    train = pd.read_csv(
        "train_triplets.txt", sep=" ", header=None, dtype=str
    )
    test = pd.read_csv(
        "test_triplets.txt", sep=" ", header=None, dtype=str
    )

    images = np.unique(
        np.append(
            np.append(
                np.append(
                    np.append(
                        np.append(train[0].values, train[1].values), train[2].values
                    ),
                    test[0].values,
                ),
                test[1].values,
            ),
            test[2].values,
        )
    )
        
    base_model = create_custom_cnn(INPUT_SIZE)
    model = create_triplet_network(base_model, INPUT_SIZE)
    model.compile(loss=triplet_loss_batch_all, optimizer="adam")
    model.fit_generator(
        generator=generate_images(train, BATCH_SIZE),
        steps_per_epoch=int(len(train[0]) / BATCH_SIZE),
        epochs=1,
        verbose=1,
    )

    embeddings = images
    embeddings_dict = {}
    for i, embedding in enumerate(embeddings):
        embeddings_dict[embedding] = base_model.predict(
            np.expand_dims(
                preprocess_input(
                    img_to_array(
                        load_img(
                            f"{BASE_RELATIVE_IMAGE_PATH + embedding}.jpg",
                            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                        )
                    )
                ),
                axis=0,
            )
        )

    y = list()
    X = list()
    for i in range(len(train[0])):
        X.append(
            np.absolute(
                embeddings_dict[train[0].values[i]]
                - embeddings_dict[train[1].values[i]]
            )
        )
        y.append(1)

        X.append(
            np.absolute(
                embeddings_dict[train[0].values[i]]
                - embeddings_dict[train[2].values[i]]
            )
        )
        y.append(0)

    train_frame = pd.DataFrame(np.vstack(X))
    model = LogisticRegression()
    model.fit(X=train_frame, y=y)
    X_pred = list()
    prediction = list()

    for i in range(len(test[0])):
        X_pred.append(
            np.absolute(
                embeddings_dict[test[0].values[i]] - embeddings_dict[test[1].values[i]]
            )
        )

        X_pred.append(
            np.absolute(
                embeddings_dict[test[0].values[i]] - embeddings_dict[test[2].values[i]]
            )
        )

    predict_frame = model.predict_proba(pd.DataFrame(np.vstack(X_pred)))[:, 1]

    for i in range(0, len(predict_frame), 2):
        if predict_frame[i] > predict_frame[i + 1]:
            prediction.append(1)
        else:
            prediction.append(0)

    pd.Series(prediction).to_csv(
        "prediction.csv", sep=",", index=False, header=False
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

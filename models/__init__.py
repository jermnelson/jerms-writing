__author__ = "Jeremy Nelson"

from tensorflow.keras.layers import Flatten, Dense  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore

from config import AIKI_NAMES, CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH


def feedforward_modal(class_names: list = CLASS_NAMES) -> Sequential:
    model = Sequential([
      # input layer
      Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
      # first hidden layer
      Dense(64, activation='relu'),
      # second hidden layer
      Dense(64, activation='relu'),
      # output layer
      Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Model Summary
    model.summary()
    return model


def aiki_feedforward() -> Sequential:
    return feedforward_modal(AIKI_NAMES)


def digits_feedforward() -> Sequential:
    digits = [i for i in range(0, 10)]
    return feedforward_modal(digits)

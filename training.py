__author__ = "Jeremy Nelson"
__license__ = "Apache 2"

import argparse
import os
import tensorflow as tf
import datetime
from PIL import Image
from tensorflow import keras
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

# Contants
TRAIN_DIR = pathlib.Path('data/train')
VALID_DIR = pathlib.Path('data/validation')

BATCH_SIZE = 100
IMG_HEIGHT, IMG_WIDTH = 56, 56
CLASS_NAMES = sorted(d.stem for d in TRAIN_DIR.glob('*'))

def feedforward_modal():
  model = keras.models.Sequential([
    # input layer
    keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    # first hidden layer
    keras.layers.Dense(64, activation='relu'),
    # output layer
    keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='categorical_crossentropy', 
                metrics=['accuracy'])


  # Model Summary
  model.summary()
  return model

def normalize_size(path):
    original = Image.open(path)
    xsize, ysize = original.size
    if xsize == 56 and ysize == 56:
        return
    center_x, center_y = int((56 - xsize)/ 2), int((56 - ysize) / 2)
    normalized = Image.new('RGB', (56,56), color='white')
    normalized.paste(original, (center_x, center_y))
    normalized.save(path, "PNG")


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
          ax = plt.subplot(5,5,n+1)
          plt.imshow(np.reshape(image_batch[n], (56,56)))
          plt.title(CLASS_NAMES[np.argwhere(label_batch[n]==1)[0][0]])
          plt.axis('off')


def get_label(file_path):
    # convert the path to a list of path components
    # The second to last is the class-directory
    return file_path.parts[-2] == np.array(CLASS_NAMES)


def decode_img(file_path):
    # load image
    img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    # convert to floats in the [0,1] range.
    img = np.float32(img) / 255.0
    # resize the image to the desired size.
    try:
        good_image = img.reshape((IMG_WIDTH, IMG_HEIGHT, 1))
        return good_image
    except:
        print(f"Failed to reshape {file_path} {sys.exc_info()}")
    

def process_path(file_path):
  label = get_label(file_path)
  img = decode_img(file_path)
  return img, label

def preprocess_images():
  for row in [TRAIN_DIR, VALID_DIR]:
    print(f"Processing all {row}")
    for class_dir in sorted(row.iterdir()):
      for img in class_dir.glob("*.png"):
        normalize_size(img)
 
if __name__ == '__main__':
  current = datetime.datetime.utcnow()
  print(f"Training Module started at {current.isoformat()}")
  parser = argparse.ArgumentParser()
  parser.add_argument("action", help="Training action")
  args = parser.parse_args()
  if args.action == 'prep':
    print("Preprocessing Images")
    preprocess_images()
  if args.action == 'run':
    print("Building Model")
    # Callbacks
    callbacks = list([
      keras.callbacks.ModelCheckpoint(
        'models/ffnn.hdf5',
        monitor='accuracy',
        mode='max',
        verbose=1
     )
    ])

    model = feedforward_modal()
    train_imgs, train_labels = zip(*(process_path(f_p) for f_p in TRAIN_DIR.glob('*/*')))
    valid_imgs, valid_labels = zip(*(process_path(f_p) for f_p in VALID_DIR.glob('*/*')))
    print("Fitting model")
    model.fit(np.array(train_imgs),
              np.array(train_labels),
              batch_size=100,
              epochs=10,
              validation_data=(np.array(valid_imgs), np.array(valid_labels)),
              callbacks=callbacks)

    #print(np.array(train_imgs).shape)
    #print(np.array(train_labels).shape)
    #print(np.array(valid_imgs).shape)
    #print(np.array(valid_labels).shape)

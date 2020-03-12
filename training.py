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

# Constants
from config import CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH
from models import feedforward_modal, aiki_feedforward

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(np.reshape(image_batch[n], (56,56)))
      plt.title(CLASS_NAMES[np.argwhere(label_batch[n]==1)[0][0]])
      plt.axis('off')


def get_label(file_path, class_names=CLASS_NAMES):
    # convert the path to a list of path components
    # The second to last is the class-directory
    return file_path.parts[-2] == np.array(class_names)


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

def train(**kwargs):
    print(f"Starting training {kwargs.get('name'. 'all')}")
    callbacks = list([
      keras.callbacks.ModelCheckpoint(
        f"models/ffnn-{kwargs.get('name', 'all')}.hdf5",
        monitor='accuracy',
        mode='max',
        verbose=1
     )
    ])

    model = kwargs.get('model', feedforward_model())
    train_imgs, train_labels = zip(*(process_path(f_p) for f_p in kwargs.get('train_paths', TRAIN_DIR.glob('*/*')))
    valid_imgs, valid_labels = zip(*(process_path(f_p) for f_p in kwargs.get('valid_paths', VALID_DIR.glob('*/*'))))
    print(f"Fitting all model")
    model.fit(np.array(train_imgs),
              np.array(train_labels),
              batch_size=100,
              epochs=10,
              validation_data=(np.array(valid_imgs), np.array(valid_labels)),
              callbacks=callbacks)

def train_aiki():
    print(f"Started training {' '.join(AIKI_NAMES)}"
    train_paths, valid_paths = [], []
    for char in AIKI_NAMES:
        for img in TRAIN_DIR.glob(f"{char}/*"):
            train_paths.append(img)
        for img in VALID_DIR.glob(f"{char}/*"):
            valid_paths.append(img) 
    train(name="aikido",
          model=aiki_feedforward, 
          train_paths=train_paths, 
          valid_paths=valid_paths)
 


def train_aiki():

if __name__ == '__main__':
  current = datetime.datetime.utcnow()
  print(f"Training Module started at {current.isoformat()}")
  parser = argparse.ArgumentParser()
  parser.add_argument("action", help="Training action")
  args = parser.parse_args()
  if args.action == 'run':
    print("Building Model")
    # Callbacks
    
    #print(np.array(train_imgs).shape)
    #print(np.array(train_labels).shape)
    #print(np.array(valid_imgs).shape)
    #print(np.array(valid_labels).shape)

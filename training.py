__author__ = "Jeremy Nelson"
__license__ = "Apache 2"

import tensorflow as tf
import datetime
from tensorflow import keras
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Contants
TRAIN_DIR = pathlib.Path('data/train')
VALID_DIR = pathlib.Path('data/validation')

BATCH_SIZE = 100
IMG_HEIGHT, IMG_WIDTH = 56, 56
CLASS_NAMES = sorted(d.stem for d in TRAIN_DIR.glob('*'))


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
  return img.reshape((IMG_WIDTH, IMG_HEIGHT, 1))

def process_path(file_path):
  label = get_label(file_path)
  img = decode_img(file_path)
  return img, label


if __name__ == '__main__':
    current = datetime.datetime.utcnow()
    print(f"Training Module started at {current.isoformat()}")


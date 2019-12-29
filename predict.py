__author__ = "Jeremy Nelson"

import cv2
import click
import numpy as np
from tensorflow import keras
from training import decode_img, normalize_size


SAMPLE_SHAPE = 56

def process_image(image_path):
  letter_img = cv2.bitwise_not(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
  print(letter_img)
  nlabels, labels, stats, ctds = cv2.connectedComponentsWithStats(letter_img)
#print(f"nlabels: {nlabels}\nlabels: {labels}\nstats: {stats}\nctds: {ctds}")
  cc_order = ctds[:0,].argsort()
  cc_order = cc_order[cc_order>0]
  cc_ids = np.array(range(nlabels))
  print(f"cc_order {cc_order} cc_ids {cc_ids} {cc_ids[cc_order]}")
  for i in cc_ids[cc_order]:
    print(f"{i} in for loop")
  return 
  char_img = np.uint8(np.where(labels == i, 255, 0))
  x, y, w, h = cv2.boundingRect(char_img)
  char_img_crop = char_img[y:y+h, x:x+w]
  top = max((SAMPLE_SHAPE - h) // 2, 0)
  bottom = max(SAMPLE_SHAPE - (h + top), 0)
  left = max((SAMPLE_SHAPE - w) // 2, 0)
  right = max(SAMPLE_SHAPE - (w + left), 0)
  print(f"cc_order: {cc_order}\ncc_ids: {cc_ids}")
  sample_img = cv2.copyMakeBorder(
    char_img_crop,
    top, bottom, left, right,
    borderType=cv2.BORDER_CONSTANT,
    value=0)
  return [sample_img]

def prep_image(image_path):
  letter_img = cv2.bitwise_not(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
  return letter_img
  #return letter_img.reshape((SAMPLE_SHAPE, SAMPLE_SHAPE, 1))/ 255.0 

@click.command()
@click.option('--model', default='ffnn.hdf5', help='Tensorflow model')
@click.option('--image', help='path to image')
def predict_image(model, image):
  classifier = keras.models.load_model(f"models/{model}")
  input_img = normalize_size(image)
  input_img = decode_img(input_img)
  classifier.predict([input_img])

if __name__ == '__main__':
  predict_image()

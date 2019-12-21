__author__ = "Jeremy Nelson"

import cv2
import click
import numpy as np
from tensorflow import keras

def process_image(image_path):
    letter_img = cv2.bitwise_not(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
    nlabels, labels, stats, ctds = cv2.connectedComponentsWithStats(letter_img)
    print(f"nlabels: {nlabels}\nlabels: {labels}\nstats: {stats}\nctds: {ctds}")
    cc_order = ctds[:0,].argsort()
    cc_order = cc_order[cc_order>0]
    cc_ids = np.array(range(nlabels))
    print(f"cc_order: {cc_order}\ncc_ids: {cc_ids}")
    return []

@click.command()
@click.option('--model', default='ffnn.hdf5', help='Tensorflow model')
@click.option('--image', help='path to image')
def predict_image(model, image):
    classifier = keras.models.load_model(f"models/{model}")
    input_img = process_image(image)
    # classifier.predict(input_img)

if __name__ == '__main__':
    predict_image()

__author__ = "Jeremy Nelson"

import datetime

from config import TRAIN_DIR, VALID_DIR
from PIL import Image

def normalize_size(path):
    original = Image.open(path)
    xsize, ysize = original.size
    if xsize == 56 and ysize == 56:
        return
    center_x, center_y = int((56 - xsize)/ 2), int((56 - ysize) / 2)
    normalized = Image.new('RGB', (56,56), color='white')
    normalized.paste(original, (center_x, center_y))
    normalized.save(path, "PNG"

def images(directories: list):
    start = datetime.datetime.utcnow()
    print(f"Starting image preparation at {start}")
    for directory in directories:
        print(f"Processing {directory}")
        for class_dir in sorted(directory.iterdir()):
            for img in class_dir.glob("*.png"):
                normalize_size(img)
    end = datetime.datetime.utcnow()
    seconds = (end-start).seconds
    print(f"Finished image preparation at {end} total time: {seconds} secs")


if __name__ == '__main__':
    print("Image preparation for Jerms Writing Project")
    images([TRAIN_DIR, VALID_DIR])
    print("Finished")

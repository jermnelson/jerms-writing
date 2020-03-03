import pathlib

TRAIN_DIR = pathlib.Path('data/train')
VALID_DIR = pathlib.Path('data/validation')


AIKI_NAMES = ["合", "気", "道"]

BATCH_SIZE = 100
IMG_HEIGHT, IMG_WIDTH = 56, 56
CLASS_NAMES = sorted(d.stem for d in TRAIN_DIR.glob('*'))

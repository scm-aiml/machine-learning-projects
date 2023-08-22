""" Configuration file
This is a configuration file designed for local machine running

Attributes:
    RANDOM_STATE (int): The random seed for any RNGs
    BATCH_SIZE (int): Batch size for training
    IMG_HEIGHT (int): Height of image in pixels
    IMG_WIDTH (int): Width of image in pixels
    DATA_FOLDER (str): Name of folder containing data
    IMG_FOLDER (str): Name of folder in data containing images
    MASKS_FOLDER (str): Name of folder in data containing masks
    TRAIN_FRACTION (float): Fraction of dataset to be used for training
"""

# RNG
RANDOM_STATE = 42

# Hyperparameters
BATCH_SIZE = 4

# Image dimensions
IMG_HEIGHT = 208
IMG_WIDTH = 304

# Folders
DATA_FOLDER = 'data'
IMG_FOLDER = 'images'
MASKS_FOLDER = 'masks'

# train-test split
TRAIN_FRACTION = 0.8

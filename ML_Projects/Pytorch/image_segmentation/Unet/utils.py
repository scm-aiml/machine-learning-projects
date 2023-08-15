""" Set of utils for UNET model in Carvana challenge

This provides a set of helper functions for use in implementing UNET
for image segmentation in the Carvana challenge (Kaggle)

Functions:
    DoubleConv: A double convolution layer with batch normalization and ReLU activation.

author: Shane Moran
(c) 2023 Shane Moran. All rights reserved.
"""

import os

def download_data():
    """ download Carvana data from Kaggle

    """
    DATA_DIR = './data'
    print(len(os.listdir(DATA_DIR)))
    # Need to download
    if  len(os.listdir(DATA_DIR)) < 3:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        api.competition_download_file('carvana-image-masking-challenge',file_name='train.zip', path=DATA_DIR, quiet = False)
        api.competition_download_file('carvana-image-masking-challenge',file_name='train_masks.zip', path=DATA_DIR, quiet = False)  
    
    # Need to unzip
    if len(os.listdir(DATA_DIR)) < 5:
        import zipfile

        # Train images
        with zipfile.ZipFile(os.path.join(DATA_DIR,'train.zip'), 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.rename("./data/train","./data/images")
        
        # Train masks
        with zipfile.ZipFile(os.path.join(DATA_DIR,'train_masks.zip'), 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.rename("./data/train_masks","./data/masks")

if __name__ == "__main__":
    
    download_data()


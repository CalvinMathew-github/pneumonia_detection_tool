import os
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from matplotlib import pyplot as plt

# Function to process and save the image
def process_and_save_image(input_path, output_path):
    # Load the image in grayscale
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Image Rescaling
    rescaled_image = cv2.resize(image, (256, 256))  # Resize to 256x256

    # Histogram Equalization
    equalized_image = cv2.equalizeHist(rescaled_image)

    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(equalized_image)

    # Convert to HSV
    hsv_image = cv2.cvtColor(cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

    # Save the processed image
    cv2.imwrite(output_path, hsv_image)

# Function to save the original image after resizing
def save_ori_image(input_path, output_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Image Rescaling
    rescaled_image = cv2.resize(image, (256, 256))
    cv2.imwrite(output_path, rescaled_image)

# Function to preprocess and save images
def preprocess_img():
    # Process and save the image in HSV format
    process_and_save_image('output1.png', 'output_hsv.png')
    process_and_save_image('output1.png', 'static/images/output_hsv.png')

    # Save the original resized image
    save_ori_image('output1.png', 'static/images/output1.png')

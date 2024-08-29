import cv2
import os
import numpy as np
from multiprocessing import Pool
import time
import sys


def analyze_lighting(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image cannot be read: {image_path}")
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(grayscale)
        return mean_brightness
    except Exception as e:
        print(f"Error processing file {image_path}: {e}")
        return None

def process_image(path):
    try:
        if path.lower().endswith((".jpg", ".png")):
            return path, analyze_lighting(path)
    except Exception as e:
        print(f"Error processing file {path}: {e}")
    return None

def sort_images_by_lighting(folder):
    files = [os.path.join(folder, filename) for filename in os.listdir(folder)]
    with Pool() as pool:
        results = pool.map(process_image, files)
    lighting_values = filter(None, results)
    sorted_images = sorted(lighting_values, key=lambda x: x[1])
    return sorted_images

if __name__ == '__main__':
    # Time the sorting by lighting process
    start_time_sort = time.time()

    # Define the folder path directly
    folder = 'lighting_test_chatGPT/photos'

    # Sort the photos by lighting
    sorted_images_with_values = sort_images_by_lighting(folder)
    for filename, value in sorted_images_with_values:
        print(f"{filename}: {value}")

    # Stop timing the sorting by lighting and print the elapsed time
    end_time_sort = time.time()
    print(f"Time taken to sort images: {end_time_sort - start_time_sort} seconds")
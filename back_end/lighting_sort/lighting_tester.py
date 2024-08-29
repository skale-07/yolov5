import cv2
import os
import sys
import numpy as np
import time

# Add the parent directory to the system path to allow imports from sibling directories
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# # Now you can import the necessary function from reduce_resolutionGLOBAL.py
# from reduce_resolutionGLOBAL.reduce_resolutionGLOBAL import reduce_resolution_in_folder

# # Assuming you want to process the photos in 'lighting_test_chatGPT/photos'
# amplification_factor = 2  # or whatever factor you need

# input_directory = 'lighting_test_chatGPT/photos'
# output_directory = 'reduce_resolutionGLOBAL/outputTestedPhotos'


# # Before running your lighting analysis, reduce the resolution of your photos
# reduce_resolution_in_folder(input_directory, output_directory, amplification_factor)




start_time = time.time()
# This function analyzes the lighting of an image
# It reads the image, converts it to grayscale, and calculates the mean brightness
def analyze_lighting(image_path):
    # Read the image from the given path
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate and return the mean brightness of the grayscale image
    mean_brightness = np.mean(grayscale)
    return mean_brightness

# This function loads photos from a specified folder and analyzes their lighting
def load_images_from_folder(folder):
    lighting_values = {}

    # Loop through each file in the folder
    for filename in os.listdir(folder):
        # Check if the file is an image (jpg or png)
        if filename.endswith(".JPG") or filename.endswith(".png") or filename.endswith(".jpg"): 
            # Construct the full file path
            path = os.path.join(folder, filename)
            # Analyze the lighting of the image and store the value
            lighting_values[path] = analyze_lighting(path) 
    return lighting_values

def extract_filename_and_value(sorted_images):
    """Extracts and returns filenames along with their lighting values."""
    filename_values = []
    for image in sorted_images:
        filename = os.path.basename(image[0])
        value = image[1]
        filename_values.append((filename, value))
    return filename_values

def sort_images_by_lighting(folder):
    lighting_values = load_images_from_folder(folder)

    # Sort the photos based on their lighting values
    sorted_images = sorted(lighting_values.items(), key=lambda x: x[1])

    # Extract filenames and their values
    sorted_filename_values = extract_filename_and_value(sorted_images)

    return sorted_filename_values

# Example usage
sorted_images_with_values = sort_images_by_lighting('reduce_resolutionGLOBAL/outputTestedPhotos')
for filename, value in sorted_images_with_values:
    print(f"{filename}: {value}")

end_time = time.time()

print(end_time - start_time) 

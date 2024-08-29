import time
import os
import cv2
from concurrent.futures import ProcessPoolExecutor



# Function to calculate average brightness of an image
def calculate_brightness(image_path, resize_width=100):
    # Read the image from disk
    img = cv2.imread(image_path)

    # Resize the image to reduce computation if it's large
    width = img.shape[1]
    if width > resize_width:
        ratio = resize_width / width
        img = cv2.resize(img, (resize_width, int(img.shape[0] * ratio)), interpolation=cv2.INTER_AREA)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the average brightness
    avg_brightness = cv2.mean(gray)[0]

    return avg_brightness

# Function to process photos in parallel and sort by brightness
def process_images_parallel_image_paths(image_paths) -> list[tuple[str, float]]:
    # image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
    
    with ProcessPoolExecutor() as executor:
        brightness_values = list(executor.map(calculate_brightness, image_paths))

    image_brightness = list(zip(image_paths, brightness_values))
    sorted_images = sorted(image_brightness, key=lambda x: x[1])

    return sorted_images

def process_images_parallel_directory(directory) -> list[tuple[str, float]]:
    image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]

    with ProcessPoolExecutor() as executor:
        brightness_values = list(executor.map(calculate_brightness, image_paths))

    image_brightness = list(zip(image_paths, brightness_values))
    sorted_images = sorted(image_brightness, key=lambda x: x[1])

    return sorted_images

    # for image_path, brightness in sorted_images:
    #     print(f"{os.path.basename(image_path)}: {brightness}")

if __name__ == '__main__':
    # Directory where photos are stored
    PHOTOS_DIR = 'Backend Programs\\test_photos'
    start_time = time.time()
    # Example usage
    process_images_parallel(PHOTOS_DIR)
    print(f"Total Time: {time.time()-start_time}")
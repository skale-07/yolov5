from PIL import Image, UnidentifiedImageError
import os

def verify_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            try:
                img = Image.open(image_path)
                img.verify()  # Verify if the image can be opened
                print(f"{filename} is a valid image.")
            except (UnidentifiedImageError, IOError) as e:
                print(f"{filename} is not a valid image. Error: {e}")

# Replace 'static\\uploads' with the path to your image directory
verify_images(r'C:\Users\tdewa\Code\snapsort_project\SnapSort\test_photos')


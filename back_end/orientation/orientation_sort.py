from PIL import Image
import os



# def sort_images_by_orientation(image_directory, sorted_directory):
#     # Create directories for sorted photos if they don't exist
#     if not os.path.exists(f'{sorted_directory}/landscape'):
#         os.makedirs(f'{sorted_directory}/landscape')
#     if not os.path.exists(f'{sorted_directory}/portrait'):
#         os.makedirs(f'{sorted_directory}/portrait')

#     # List all files in the directory
#     for filename in os.listdir(image_directory):
#         if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#             continue  # Skip files that are not photos

#         # Open the image
#         with Image.open(os.path.join(image_directory, filename)) as img:
#             # Check the orientation based on width and height
#             if img.width > img.height:
#                 # Image is landscape
#                 orientation = 'landscape'
#             else:
#                 # Image is portrait
#                 orientation = 'portrait'

#             # Move the image to the corresponding directory
#             img.save(os.path.join(sorted_directory, orientation, filename))




# def sort_images_by_orientation(image_directory, sorted_directory, desired_orientation):
#     # Create directories for sorted photos if they don't exist
#     if not os.path.exists(f'{sorted_directory}/{desired_orientation}'):
#         os.makedirs(f'{sorted_directory}/{desired_orientation}')

#     # List all files in the directory
#     for filename in os.listdir(image_directory):
#         if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#             continue  # Skip files that are not photos

#         # Open the image
#         with Image.open(os.path.join(image_directory, filename)) as img:
#             # Check the orientation based on width and height
#             if img.width > img.height:
#                 # Image is landscape
#                 orientation = 'landscape'
#             else:
#                 # Image is portrait
#                 orientation = 'portrait'

#             # Move the image to the corresponding directory only if it matches the desired orientation
#             if orientation == desired_orientation:
#                 img.save(os.path.join(sorted_directory, orientation, filename))

def sort_images_by_orientation(image_directory, desired_orientation) -> list[str]:
    # Ensure the desired_orientation is either 'portrait' or 'landscape'
    if desired_orientation not in ['portrait', 'landscape']:
        raise ValueError("desired_orientation must be 'portrait' or 'landscape'")
    
    # List to store paths of photos that match the desired orientation
    sorted_image_paths = []
    
    # Iterate over all files in the given directory
    for filename in os.listdir(image_directory):
        # Construct the absolute path of the file
        file_path = os.path.join(image_directory, filename)
        
        # Try to open the image file
        try:
            with Image.open(file_path) as img:
                # Determine the orientation of the image
                if img.width > img.height and desired_orientation == 'landscape':
                    sorted_image_paths.append(file_path)
                elif img.height > img.width and desired_orientation == 'portrait':
                    sorted_image_paths.append(file_path)
        except IOError:
            # If the file cannot be opened as an image, skip it
            continue
    
    return sorted_image_paths

if __name__ == '__main__':
    # # Directory containing the photos
    # image_directory = 'test_photos'
    # sorted_directory = 'Backend_Programs/orientation/output'

    # # Run the sorting function
    # sort_images_by_orientation(image_directory, sorted_directory)

    print(sort_images_by_orientation('test_photos', 'portrait'))

    print("Images have been sorted by orientation.")

# import torch
# from PIL import Image
# import os
#
# def detect_objects(image_path):
#     # Load YOLOv5 model from PyTorch Hub
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#     # Load image
#     img = Image.open(image_path)
#     # Perform inference
#     results = model(img)
#     # Get the Pandas DataFrame with detected objects
#     detected_objects = results.pandas().xyxy[0]  # Pandas DataFrame with detected objects
#     return set(detected_objects['name'])  # Return a set of unique detected object names
#
# if __name__ == '__main__':
#     image_dir = 'back_end/object_detection/photos'
#     for filename in os.listdir(image_dir):
#         if filename.endswith(('.png', '.jpg', '.jpeg')):
#             image_path = os.path.join(image_dir, filename)
#             detected_objects = detect_objects(image_path)
#             print(f"Detected objects in {filename}: {detected_objects}")


import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil


# def detect_objects(image_path):
#
#     # Create the output directory if it doesn't exist
#     # os.makedirs(output_dir, exist_ok=True)
#
#     # Load YOLOv5 model from PyTorch Hub
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#
#     # Load image
#     img = Image.open(image_path)
#
#     # Perform inference
#     results = model(img)
#
#     # Print results
#     results.print()  # Print results to console
#
#     # Get the Pandas DataFrame with detected objects
#     detected_objects = results.pandas().xyxy[0]  # Pandas DataFrame with detected objects
#     print(detected_objects)
#
#     # Create a dictionary to store the file path and detected object names
#     detected_objects_dict = {image_path: set(detected_objects['name'])}
#
#     return detected_objects['name']
#
#
#     # CODE FOR STORING THE OUTPUT IMAGE WITH BOUNDING BOXES
#
#     # print(detected_objects_dict)
#     #
#     # # Save image with bounding boxes to a temporary directory
#     # temp_dir = 'runs/detect/temp'
#     # results.save(save_dir=temp_dir)
#     #
#     # # Move the saved image to the output directory with its original name
#     # image_filename = os.path.basename(image_path)
#     # temp_image_path = os.path.join(temp_dir, image_filename)
#     # final_image_path = os.path.join(output_dir, image_filename)
#     # shutil.move(temp_image_path, final_image_path)
#     #
#     # # Clean up the temporary directory
#     # shutil.rmtree(temp_dir)
#     #
#     # # Check if the file exists and display the image with bounding boxes
#     # if os.path.exists(final_image_path):
#     #     result_img = Image.open(final_image_path)
#     #     # plt.imshow(result_img)
#     #     # plt.axis('off')
#     #     # plt.show()
#     # else:
#     #     print(f"File not found: {final_image_path}")


# NEW FUNCTION TO INTEGRATE WITH BLUR FOR FRONT END
def detect_objects(image_path):
    # Load YOLOv5 model from PyTorch Hub
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Load image
    img = Image.open(image_path)

    # Perform inference
    results = model(img)

    # Print results
    results.print()  # Print results to console

    # Get the Pandas DataFrame with detected objects
    detected_objects = results.pandas().xyxy[0]  # Pandas DataFrame with detected objects
    print(detected_objects)

    return detected_objects



if __name__ == '__main__':
    image_path = r'C:\Users\tdewa\Code\snapsort_project\SnapSort\back_end\object_detection\images\bus.jpg'  # Ensure this path matches your image location

    detect_objects(image_path)

    # Define the output directory
    # output_dir = r'C:\Users\tdewa\Code\snapsort_project\SnapSort\back_end\object_detection\runs\detect\exp'
    #
    # detect_objects(image_path)
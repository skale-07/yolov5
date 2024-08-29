### DETECT MAIN SUBJECT BY USING CONFIDENCE LEVEL OF DETECTED OBJECTS

## VERSION 1
# import torch
# from PIL import Image
# import numpy as np
# import imutils
# import cv2
# import matplotlib.pyplot as plt
# import os
#
# def detect_objects(image_path):
#     # Load YOLOv5 model from PyTorch Hub
#     print(f"Loading model for image: {image_path}")
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#
#     # Load image
#     img = Image.open(image_path)
#
#     # Perform inference
#     results = model(img)
#
#     # Get the Pandas DataFrame with detected objects
#     detected_objects = results.pandas().xyxy[0]
#
#     # Find the main subject based on the highest confidence score
#     if len(detected_objects) > 0:
#         main_subject = detected_objects.iloc[detected_objects['confidence'].idxmax()]
#         bbox = (int(main_subject['xmin']), int(main_subject['ymin']),
#                 int(main_subject['xmax']), int(main_subject['ymax']))
#         print(f"Detected main subject with bbox: {bbox}")
#         return bbox
#     else:
#         print("No objects detected.")
#         return None
#
# def detect_blur_fft(image, size=60, thresh=10, vis=False):
#     (h, w) = image.shape
#     (cX, cY) = (int(w / 2.0), int(h / 2.0))
#
#     try:
#         fft = np.fft.fft2(image)
#         fftShift = np.fft.fftshift(fft)
#
#         if vis:
#             magnitude = 20 * np.log(np.abs(fftShift))
#             (fig, ax) = plt.subplots(1, 2, )
#             ax[0].imshow(image, cmap="gray")
#             ax[0].set_title("Input")
#             ax[0].set_xticks([])
#             ax[0].set_yticks([])
#             ax[1].imshow(magnitude, cmap="gray")
#             ax[1].set_title("Magnitude Spectrum")
#             ax[1].set_xticks([])
#             ax[1].set_yticks([])
#             plt.show()
#
#         fftShift[cY - size:cY + size, cX - size:cX + size] = 0
#         fftShift = np.fft.ifftshift(fftShift)
#         recon = np.fft.ifft2(fftShift)
#         magnitude = 20 * np.log(np.abs(recon))
#         mean = np.mean(magnitude)
#         return mean, mean <= thresh
#     except ValueError as e:
#         print(f"Error in FFT processing: {e}")
#         return None, False
#
# def blur_detection(image_path, thresh):
#     orig = cv2.imread(image_path)
#     orig = imutils.resize(orig, width=500)
#     gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
#
#     print(f"Processing image: {image_path}")
#     bbox = detect_objects(image_path)
#     if bbox is not None:
#         x_min, y_min, x_max, y_max = bbox
#         roi = gray[y_min:y_max, x_min:x_max]
#         if roi.size == 0:
#             print(f"Warning: ROI is empty for {image_path}. Using full image.")
#             roi = gray
#     else:
#         roi = gray
#
#     (mean, blurry) = detect_blur_fft(roi, size=60, thresh=thresh)
#     if mean is None:
#         return None
#
#     print(f"Blur score: {mean}, Blurry: {blurry}")
#
#     # Display the original image, ROI, and blur detection result
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
#     plt.title("Original Image")
#
#     if bbox is not None:
#         cv2.rectangle(orig, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#         plt.subplot(1, 2, 2)
#         plt.imshow(roi, cmap="gray")
#         plt.title(f"ROI Blur Score: {mean}")
#
#     plt.show()
#
#     return mean
#
# def main(directory_path, thresh=20, ascending=False):
#     blur_scores = []
#
#     for filename in os.listdir(directory_path):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             image_path = os.path.join(directory_path, filename)
#             blur_score = blur_detection(image_path, thresh)
#             if blur_score is not None:
#                 blur_scores.append((blur_score, image_path))
#
#     blur_scores.sort(key=lambda x: x[0], reverse=not ascending)
#     sorted_image_paths = [image_path for _, image_path in blur_scores]
#     return sorted_image_paths
#
# if __name__ == '__main__':
#     directory_path = r"C:\Users\tdewa\Code\snapsort_project\SnapSort\back_end\blur\sample_photos"
#     sorted_images = main(directory_path, thresh=20)
#     print("Sorted photos by blur score:")
#     for image_path in sorted_images:
#         print(image_path)



# ## Version 2 - Fixed ROI Is empty issue
#
# import torch
# from PIL import Image
# import numpy as np
# import imutils
# import cv2
# import matplotlib.pyplot as plt
# import os
#
# def detect_objects(image_path, confidence_thresh):
#     # Load YOLOv5 model from PyTorch Hub
#     print(f"Loading model for image: {image_path}")
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#
#     # Load image
#     img = Image.open(image_path)
#
#     # Perform inference
#     results = model(img)
#
#     # Get the Pandas DataFrame with detected objects
#     detected_objects = results.pandas().xyxy[0]
#
#     # Filter objects by confidence threshold
#     filtered_objects = detected_objects[detected_objects['confidence'] >= confidence_thresh]
#
#     # Find the main subject based on the highest confidence score
#     if len(filtered_objects) > 0:
#         main_subject = filtered_objects.iloc[filtered_objects['confidence'].idxmax()]
#         bbox = (int(main_subject['xmin']), int(main_subject['ymin']),
#                 int(main_subject['xmax']), int(main_subject['ymax']))
#         confidence = main_subject['confidence']
#         name = main_subject['name']
#         print(f"Detected main subject: {name} with confidence: {confidence:.2f} and bbox: {bbox}")
#         return bbox, confidence, name
#     else:
#         print("No objects detected with confidence above threshold.")
#         return None, None, None
#
# def detect_blur_fft(image, size=60, thresh=10, vis=False):
#     (h, w) = image.shape
#     (cX, cY) = (int(w / 2.0), int(h / 2.0))
#
#     try:
#         fft = np.fft.fft2(image)
#         fftShift = np.fft.fftshift(fft)
#
#         if vis:
#             magnitude = 20 * np.log(np.abs(fftShift))
#             (fig, ax) = plt.subplots(1, 2, )
#             ax[0].imshow(image, cmap="gray")
#             ax[0].set_title("Input")
#             ax[0].set_xticks([])
#             ax[0].set_yticks([])
#             ax[1].imshow(magnitude, cmap="gray")
#             ax[1].set_title("Magnitude Spectrum")
#             ax[1].set_xticks([])
#             ax[1].set_yticks([])
#             plt.show()
#
#         fftShift[cY - size:cY + size, cX - size:cX + size] = 0
#         fftShift = np.fft.ifftshift(fftShift)
#         recon = np.fft.ifft2(fftShift)
#         magnitude = 20 * np.log(np.abs(recon))
#         mean = np.mean(magnitude)
#         return mean, mean <= thresh
#     except ValueError as e:
#         print(f"Error in FFT processing: {e}")
#         return None, False
#
# def blur_detection(image_path, thresh, confidence_thresh):
#     orig = cv2.imread(image_path)
#     orig = imutils.resize(orig, width=500)
#     gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
#
#     print(f"Processing image: {image_path}")
#     bbox, confidence, name = detect_objects(image_path, confidence_thresh)
#     if bbox is not None:
#         x_min, y_min, x_max, y_max = bbox
#         print(f"Bounding box coordinates: {bbox}")
#
#         # Ensure the bounding box coordinates are within image bounds
#         x_min = max(0, x_min)
#         y_min = max(0, y_min)
#         x_max = min(gray.shape[1], x_max)
#         y_max = min(gray.shape[0], y_max)
#
#         roi = gray[y_min:y_max, x_min:x_max]
#         if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
#             print(f"Warning: ROI is empty or invalid for {image_path}. Using full image.")
#             roi = gray
#     else:
#         print(f"No valid main subject detected with confidence above {confidence_thresh}. Using full image.")
#         roi = gray
#
#     (mean, blurry) = detect_blur_fft(roi, size=60, thresh=thresh)
#     if mean is None:
#         return None
#
#     print(f"Blur score: {mean}, Blurry: {blurry}")
#
#     # Display the original image, ROI, and blur detection result
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
#     plt.title("Original Image")
#
#     if bbox is not None:
#         cv2.rectangle(orig, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#         plt.subplot(1, 2, 2)
#         plt.imshow(roi, cmap="gray")
#         plt.title(f"ROI Blur Score: {mean}")
#
#     plt.show()
#
#     return mean
#
# def main(directory_path, thresh=20, confidence_thresh=0.5, ascending=False):
#     blur_scores = []
#
#     for filename in os.listdir(directory_path):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             image_path = os.path.join(directory_path, filename)
#             blur_score = blur_detection(image_path, thresh, confidence_thresh)
#             if blur_score is not None:
#                 blur_scores.append((blur_score, image_path))
#
#     blur_scores.sort(key=lambda x: x[0], reverse=not ascending)
#     sorted_image_paths = [image_path for _, image_path in blur_scores]
#     return sorted_image_paths
#
# if __name__ == '__main__':
#     directory_path = r"C:\Users\tdewa\Code\snapsort_project\SnapSort\back_end\blur\sample_photos"
#     confidence_thresh = 0.6  # Set your desired confidence threshold here
#     sorted_images = main(directory_path, thresh=20, confidence_thresh=confidence_thresh)
#     print("Sorted photos by blur score:")
#     for image_path in sorted_images:
#         print(image_path)






## Version 3 - INTEGRATION WITH FRONT END


import torch
from PIL import Image
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import os

def detect_objects(image_path, confidence_thresh):
    # Load YOLOv5 model from PyTorch Hub
    print(f"Loading model for image: {image_path}")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Load image
    img = Image.open(image_path)

    # Perform inference
    results = model(img)

    # Get the Pandas DataFrame with detected objects
    detected_objects = results.pandas().xyxy[0]

    # Filter objects by confidence threshold
    filtered_objects = detected_objects[detected_objects['confidence'] >= confidence_thresh]

    return filtered_objects

def detect_blur_fft(image, size=60, thresh=10, vis=False):
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    try:
        fft = np.fft.fft2(image)
        fftShift = np.fft.fftshift(fft)

        if vis:
            magnitude = 20 * np.log(np.abs(fftShift))
            (fig, ax) = plt.subplots(1, 2, )
            ax[0].imshow(image, cmap="gray")
            ax[0].set_title("Input")
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].imshow(magnitude, cmap="gray")
            ax[1].set_title("Magnitude Spectrum")
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            plt.show()

        fftShift[cY - size:cY + size, cX - size:cX + size] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)
        return mean, mean <= thresh
    except ValueError as e:
        print(f"Error in FFT processing: {e}")
        return None, False

def calculate_blurriness(image_path, detected_bboxes, thresh=20, confidence_thresh=0.5):
    orig = cv2.imread(image_path)
    h_orig, w_orig = orig.shape[:2]
    orig = imutils.resize(orig, width=500)
    h_resized, w_resized = orig.shape[:2]
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    print(f"Processing image: {image_path}")

    if not detected_bboxes.empty:
        main_subject = detected_bboxes.iloc[0]
        x_min, y_min, x_max, y_max = int(main_subject['xmin']), int(main_subject['ymin']), int(main_subject['xmax']), int(main_subject['ymax'])
        print(f"Original bounding box coordinates: {(x_min, y_min, x_max, y_max)}")

        # Scale bounding box coordinates to the resized image dimensions
        x_min = int(x_min * (w_resized / w_orig))
        y_min = int(y_min * (h_resized / h_orig))
        x_max = int(x_max * (w_resized / w_orig))
        y_max = int(y_max * (h_resized / h_orig))
        scaled_bbox = (x_min, y_min, x_max, y_max)
        print(f"Scaled bounding box coordinates: {scaled_bbox}")

        # Ensure the bounding box coordinates are within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(gray.shape[1], x_max)
        y_max = min(gray.shape[0], y_max)

        roi = gray[y_min:y_max, x_min:x_max]
        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            print(f"Warning: ROI is empty or invalid for {image_path}. Using full image.")
            roi = gray
    else:
        print(f"No valid main subject detected with confidence above {confidence_thresh}. Using full image.")
        roi = gray

    (mean, blurry) = detect_blur_fft(roi, size=60, thresh=thresh)
    if mean is None:
        return None

    print(f"Blur score: {mean}, Blurry: {blurry}")

    return mean

def main(directory_path, thresh=20, confidence_thresh=0.5, ascending=False):
    blur_scores = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
            detected_bboxes = detect_objects(image_path, confidence_thresh)
            blur_score = calculate_blurriness(image_path, detected_bboxes, thresh, confidence_thresh)
            if blur_score is not None:
                blur_scores.append((blur_score, image_path))

    blur_scores.sort(key=lambda x: x[0], reverse=not ascending)
    sorted_image_paths = [image_path for _, image_path in blur_scores]
    return sorted_image_paths

if __name__ == '__main__':
    directory_path = r"C:\Users\tdewa\Code\snapsort_project\SnapSort\back_end\blur\sample_photos"
    confidence_thresh = 0.6  # Set your desired confidence threshold here
    sorted_images = main(directory_path, thresh=20, confidence_thresh=confidence_thresh)
    print("Sorted photos by blur score:")
    for image_path in sorted_images:
        print(image_path)













# ### DETECT MAIN SUBJECT BY USING PERCENTAGE AREA OF DETECTED OBJECTS
#
# import torch
# from PIL import Image
# import numpy as np
# import imutils
# import cv2
# import matplotlib.pyplot as plt
# import os
#
# def detect_objects(image_path):
#     # Load YOLOv5 model from PyTorch Hub
#     print(f"Loading model for image: {image_path}")
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#
#     # Load image
#     img = Image.open(image_path)
#
#     # Perform inference
#     results = model(img)
#
#     # Get the Pandas DataFrame with detected objects
#     detected_objects = results.pandas().xyxy[0]
#
#     return detected_objects
#
# def get_main_subject(image_path, min_area_percentage=0.5):
#     detected_objects = detect_objects(image_path)
#
#     # Calculate the total area of the image
#     image = Image.open(image_path)
#     image_area = image.width * image.height
#
#     # Print details of all detected objects
#     print(f"Detected {len(detected_objects)} objects in the image.")
#     for _, row in detected_objects.iterrows():
#         object_area = (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin'])
#         area_percentage = object_area / image_area * 100
#         print(f"Object: {row['name']}, Confidence: {row['confidence']:.2f}, Area: {object_area} ({area_percentage:.2f}%)")
#
#     # Check if any detected object meets the area threshold
#     for _, row in detected_objects.iterrows():
#         object_area = (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin'])
#         if object_area / image_area >= min_area_percentage:
#             bbox = (int(row['xmin']), int(row['ymin']),
#                     int(row['xmax']), int(row['ymax']))
#             print(f"Detected main subject with bbox: {bbox}")
#             return bbox
#
#     print("No main subject detected based on area criteria.")
#     return None
#
# def detect_blur_fft(image, size=60, thresh=10, vis=False):
#     (h, w) = image.shape
#     (cX, cY) = (int(w / 2.0), int(h / 2.0))
#
#     try:
#         fft = np.fft.fft2(image)
#         fftShift = np.fft.fftshift(fft)
#
#         if vis:
#             magnitude = 20 * np.log(np.abs(fftShift))
#             (fig, ax) = plt.subplots(1, 2, )
#             ax[0].imshow(image, cmap="gray")
#             ax[0].set_title("Input")
#             ax[0].set_xticks([])
#             ax[0].set_yticks([])
#             ax[1].imshow(magnitude, cmap="gray")
#             ax[1].set_title("Magnitude Spectrum")
#             ax[1].set_xticks([])
#             ax[1].set_yticks([])
#             plt.show()
#
#         fftShift[cY - size:cY + size, cX - size:cX + size] = 0
#         fftShift = np.fft.ifftshift(fftShift)
#         recon = np.fft.ifft2(fftShift)
#         magnitude = 20 * np.log(np.abs(recon))
#         mean = np.mean(magnitude)
#         return mean, mean <= thresh
#     except ValueError as e:
#         print(f"Error in FFT processing: {e}")
#         return None, False
#
# def blur_detection(image_path, thresh, min_area_percentage):
#     orig = cv2.imread(image_path)
#     orig = imutils.resize(orig, width=500)
#     gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
#
#     print(f"Processing image: {image_path}")
#     bbox = get_main_subject(image_path, min_area_percentage)
#     if bbox is not None:
#         x_min, y_min, x_max, y_max = bbox
#         roi = gray[y_min:y_max, x_min:x_max]
#         if roi.size == 0:
#             print(f"Warning: ROI is empty for {image_path}. Using full image.")
#             roi = gray
#     else:
#         roi = gray
#
#     (mean, blurry) = detect_blur_fft(roi, size=60, thresh=thresh)
#     if mean is None:
#         return None
#
#     print(f"Blur score: {mean}, Blurry: {blurry}")
#
#     # Display the original image, ROI, and blur detection result
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
#     plt.title("Original Image")
#
#     if bbox is not None:
#         cv2.rectangle(orig, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#         plt.subplot(1, 2, 2)
#         plt.imshow(roi, cmap="gray")
#         plt.title(f"ROI Blur Score: {mean}")
#
#     plt.show()
#
#     return mean
#
# def main(directory_path, thresh=20, min_area_percentage=0.5, ascending=False):
#     blur_scores = []
#
#     for filename in os.listdir(directory_path):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             image_path = os.path.join(directory_path, filename)
#             blur_score = blur_detection(image_path, thresh, min_area_percentage)
#             if blur_score is not None:
#                 blur_scores.append((blur_score, image_path))
#
#     blur_scores.sort(key=lambda x: x[0], reverse=not ascending)
#     sorted_image_paths = [image_path for _, image_path in blur_scores]
#     return sorted_image_paths
#
# if __name__ == '__main__':
#     directory_path = r"C:\Users\tdewa\Code\snapsort_project\SnapSort\back_end\blur\sample_photos"
#     sorted_images = main(directory_path, thresh=20, min_area_percentage=0.5)
#     print("Sorted photos by blur score:")
#     for image_path in sorted_images:
#         print(image_path)

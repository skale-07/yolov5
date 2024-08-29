# import os
# import face_recognition
# import cv2
# import json
# import numpy as np
#
# ENCODINGS_FILE = 'face_encodings_main.json'
#
#
# def save_encodings(encodings, file_path):
#     with open(file_path, 'w') as file:
#         json_encodings = [(path, encoding.tolist()) for path, encoding in encodings]
#         json.dump(json_encodings, file)
#
#
# def load_encodings(file_path):
#     if os.path.exists(file_path):
#         try:
#             with open(file_path, 'r') as file:
#                 json_encodings = json.load(file)
#                 encodings = [(path, np.array(encoding)) for path, encoding in json_encodings]
#                 return encodings
#         except json.JSONDecodeError:
#             print(f"Error: JSON decoding failed for file {file_path}. File might be empty or corrupted.")
#             return []
#     return []
#
#
# def load_and_encode_faces_from_directory(directory, resize_factor=0.25):
#     face_encodings = load_encodings(ENCODINGS_FILE)
#     processed_files = {path for path, _ in face_encodings}
#
#     print(f"Loading images from directory: {directory}")
#     for filename in os.listdir(directory):
#         if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#             path = os.path.join(directory, filename)
#             if path in processed_files:
#                 print(f"Skipping already processed file: {path}")
#                 continue
#
#             print(f"Processing file: {path}")
#             image = face_recognition.load_image_file(path)
#             small_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
#             encodings = face_recognition.face_encodings(small_image)
#             if encodings:
#                 for encoding in encodings:
#                     face_encodings.append((path, encoding))
#                 print(f"Encoded {len(encodings)} faces from {path}")
#             else:
#                 print(f"No faces found in {path}")
#
#     save_encodings(face_encodings, ENCODINGS_FILE)
#     print(f"Total faces encoded: {len(face_encodings)}")
#     return face_encodings
#
#
# def extract_faces_from_image(image_path, resize_factor=0.25):
#     print(f"Extracting faces from reference image: {image_path}")
#     image = face_recognition.load_image_file(image_path)
#     small_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
#     face_locations = face_recognition.face_locations(small_image)
#     face_encodings = face_recognition.face_encodings(small_image, face_locations)
#     print(f"Found {len(face_encodings)} faces in reference image")
#     return face_encodings, face_locations
#
#
# def display_faces_for_selection(image_path):
#     face_encodings, face_locations = extract_faces_from_image(image_path)
#     # Display faces using OpenCV or another method (UI implementation needed here)
#     # This part will depend on your front-end framework
#     return face_encodings
#
#
# def find_matching_faces(known_faces, selected_face_encoding, tolerance=0.6):
#     matches = []
#     print(f"Matching selected face against {len(known_faces)} known faces")
#     for path, encoding in known_faces:
#         match = face_recognition.compare_faces([encoding], selected_face_encoding, tolerance=tolerance)
#         if match[0]:
#             matches.append(path)
#             print(f"Match found: {path}")
#     print(f"Total matches found: {len(matches)}")
#     return matches
#
#
# def sort_photos_by_selected_face(directory, reference_image_path, selected_face_index, resize_factor=0.25):
#     known_faces = load_and_encode_faces_from_directory(directory, resize_factor)
#     reference_face_encodings = display_faces_for_selection(reference_image_path)
#
#     if len(reference_face_encodings) <= selected_face_index:
#         print(
#             f"Error: Selected face index {selected_face_index} out of range. Only {len(reference_face_encodings)} faces found.")
#         return []
#
#     selected_face_encoding = reference_face_encodings[selected_face_index]
#     matching_faces = find_matching_faces(known_faces, selected_face_encoding)
#     return matching_faces
#
#
# # Example usage
# directory = 'photos'
# reference_image_path = r"C:\Users\tdewa\Code\snapsort_project\SnapSort\back_end\facial_detection\cropped_output\IMG_7988_face_1.jpg"
# selected_face_index = 0  # Assume the user selects the first face displayed
#
# sorted_images = sort_photos_by_selected_face(directory, reference_image_path, selected_face_index)
# print("Sorted images containing the selected face:")
# print(sorted_images)





################# VERSION 2 ########################

# import os
# import face_recognition
# import cv2
# import json
# import numpy as np
# from collections import defaultdict
#
# ENCODINGS_FILE = 'face_encodings.json'
# REFERENCE_IMAGES_DIR = 'reference_images'
#
#
# def save_encodings(encodings, file_path):
#     with open(file_path, 'w') as file:
#         json_encodings = [(path, encoding.tolist() if isinstance(encoding, np.ndarray) else encoding) for path, encoding
#                           in encodings]
#         json.dump(json_encodings, file)
#
#
# def load_encodings(file_path):
#     if os.path.exists(file_path):
#         try:
#             with open(file_path, 'r') as file:
#                 json_encodings = json.load(file)
#                 encodings = [(path, np.array(encoding) if isinstance(encoding, list) else encoding) for path, encoding
#                              in json_encodings]
#                 return encodings
#         except json.JSONDecodeError:
#             print(f"Error: JSON decoding failed for file {file_path}. File might be empty or corrupted.")
#             return []
#     return []
#
#
# def load_and_encode_faces_from_directory(directory, resize_factor=0.25):
#     face_encodings = load_encodings(ENCODINGS_FILE)
#     processed_files = {path for path, _ in face_encodings}
#
#     print(f"Loading images from directory: {directory}")
#     for filename in os.listdir(directory):
#         if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#             path = os.path.join(directory, filename)
#             if path in processed_files:
#                 print(f"Skipping already processed file: {path}")
#                 continue
#
#             print(f"Processing file: {path}")
#             image = face_recognition.load_image_file(path)
#             small_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
#             encodings = face_recognition.face_encodings(small_image)
#             if encodings:
#                 for encoding in encodings:
#                     face_encodings.append((path, encoding))
#                 print(f"Encoded {len(encodings)} faces from {path}")
#             else:
#                 print(f"No faces found in {path}")
#                 face_encodings.append((path, None))
#
#     save_encodings(face_encodings, ENCODINGS_FILE)
#     print(f"Total faces encoded: {len(face_encodings)}")
#     return face_encodings
#
#
# def extract_faces_from_image(image_path, resize_factor=0.25):
#     image = face_recognition.load_image_file(image_path)
#     small_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
#     face_locations = face_recognition.face_locations(small_image)
#     face_encodings = face_recognition.face_encodings(small_image, face_locations)
#     return face_encodings, face_locations
#
#
# def save_reference_image(image, face_location, output_dir, filename, resize_factor):
#     top, right, bottom, left = face_location
#     top = int(top / resize_factor)
#     right = int(right / resize_factor)
#     bottom = int(bottom / resize_factor)
#     left = int(left / resize_factor)
#     face_image = image[top:bottom, left:right]
#     face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     face_path = os.path.join(output_dir, filename)
#     cv2.imwrite(face_path, face_image_bgr)
#     return face_path
#
#
# def create_reference_images(known_faces, output_dir=REFERENCE_IMAGES_DIR, resize_factor=0.25):
#     reference_encodings = []
#     reference_images = []
#     seen_faces = set()
#     for path, encoding in known_faces:
#         if encoding is None or any(
#                 face_recognition.compare_faces([ref_enc for ref_path, ref_enc in reference_encodings], encoding)):
#             continue
#         image = face_recognition.load_image_file(path)
#         small_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
#         face_locations = face_recognition.face_locations(small_image)
#         for i, face_location in enumerate(face_locations):
#             current_encoding = face_recognition.face_encodings(small_image, [face_location])[0]
#             if face_recognition.compare_faces([current_encoding], encoding)[0]:
#                 reference_path = save_reference_image(image, face_location, output_dir,
#                                                       f'reference_face_{len(reference_encodings)}.jpg', resize_factor)
#                 reference_encodings.append((reference_path, current_encoding))
#                 reference_images.append(reference_path)
#                 seen_faces.add(path)
#                 break
#     return reference_encodings, reference_images
#
#
# def find_matching_faces(known_faces, selected_face_encoding, tolerance=0.55):
#     matches = []
#     for path, encoding in known_faces:
#         if encoding is None:
#             continue
#         match = face_recognition.compare_faces([encoding], selected_face_encoding, tolerance=tolerance)
#         if match[0]:
#             matches.append(path)
#     return matches
#
#
# def sort_photos_by_selected_face(directory, selected_face_index, resize_factor=0.25):
#     known_faces = load_and_encode_faces_from_directory(directory, resize_factor)
#     reference_faces, reference_images = create_reference_images(known_faces, resize_factor=resize_factor)
#
#     if len(reference_faces) <= selected_face_index:
#         print(
#             f"Error: Selected face index {selected_face_index} out of range. Only {len(reference_faces)} reference faces found.")
#         return []
#
#     selected_face_encoding = reference_faces[selected_face_index][1]
#     matching_faces = find_matching_faces(known_faces, selected_face_encoding)
#     return matching_faces
#
#
# # Example usage
# directory = 'photos'
#
# # Simulate user selection of a reference face
# reference_encodings, reference_images = create_reference_images(load_and_encode_faces_from_directory(directory))
# print("Reference faces available for selection:")
# for i, reference_image in enumerate(reference_images):
#     print(f"{i}: {reference_image}")
#
# # Assuming the user selects a face by index (for example, index 0)
# selected_face_index = 1  # Replace with user input in a real application
# sorted_images = sort_photos_by_selected_face(directory, selected_face_index)
# print("Sorted images containing the selected face:")
# print(sorted_images)



############### VERSION 3 ##########l############

import os
import face_recognition
import cv2
import json
import numpy as np
from collections import defaultdict

ENCODINGS_FILE = r'C:\Users\tdewa\Code\snapsort_project\SnapSort - GitHub Deployment Copy\snapsort-deployment-final\static\face_encodings.json'
REFERENCE_IMAGES_DIR = r'C:\Users\tdewa\Code\snapsort_project\SnapSort - GitHub Deployment Copy\snapsort-deployment-final\static\reference_images'


def save_encodings(encodings, file_path):
    print("\nSaving encodings to file...")
    with open(file_path, 'w') as file:
        json_encodings = [(path, encoding.tolist() if isinstance(encoding, np.ndarray) else encoding) for path, encoding
                          in encodings]
        json.dump(json_encodings, file)
    print("Encodings saved successfully.\n")


def load_encodings(file_path):
    print("\nLoading encodings from file...")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                json_encodings = json.load(file)
                encodings = [(path, np.array(encoding) if isinstance(encoding, list) else encoding) for path, encoding
                             in json_encodings]
                print("Encodings loaded successfully.\n")
                return encodings
        except json.JSONDecodeError:
            print(f"Error: JSON decoding failed for file {file_path}. File might be empty or corrupted.\n")
            return []
    print("No existing encodings file found.\n")
    return []


def load_and_encode_faces_from_directory(directory, resize_factor=0.25):
    print("\nLoading and encoding faces from directory...")
    face_encodings = load_encodings(ENCODINGS_FILE)
    processed_files = {path for path, _ in face_encodings}

    print(f"Scanning directory: {directory}")
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(directory, filename)
            if path in processed_files:
                print(f"Skipping already processed file: {path}")
                continue

            print(f"Processing file: {path}")
            image = face_recognition.load_image_file(path)
            small_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
            encodings = face_recognition.face_encodings(small_image)
            if encodings:
                for encoding in encodings:
                    face_encodings.append((path, encoding))
                print(f"Encoded {len(encodings)} faces from {path}")
            else:
                print(f"No faces found in {path}")
                face_encodings.append((path, None))

    save_encodings(face_encodings, ENCODINGS_FILE)
    print(f"Total faces encoded: {len(face_encodings)}\n")
    return face_encodings


def load_and_encode_faces_from_image_paths(image_paths, resize_factor=0.25):
    print("\nLoading and encoding faces from image paths...")
    face_encodings = load_encodings(ENCODINGS_FILE)
    processed_files = {path for path, _ in face_encodings}

    print(f"Processing image paths...")
    for path in image_paths:
        if path in processed_files:
            print(f"Skipping already processed file: {path}")
            continue

        print(f"Processing file: {path}")
        image = face_recognition.load_image_file(path)
        small_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
        encodings = face_recognition.face_encodings(small_image)
        if encodings:
            for encoding in encodings:
                face_encodings.append((path, encoding))
            print(f"Encoded {len(encodings)} faces from {path}")
        else:
            print(f"No faces found in {path}")
            face_encodings.append((path, None))

    save_encodings(face_encodings, ENCODINGS_FILE)
    print(f"Total faces encoded: {len(face_encodings)}\n")
    return face_encodings


def extract_faces_from_image(image_path, resize_factor=0.25):
    print(f"Extracting faces from image: {image_path}")
    image = face_recognition.load_image_file(image_path)
    small_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
    face_locations = face_recognition.face_locations(small_image)
    face_encodings = face_recognition.face_encodings(small_image, face_locations)
    print(f"Found {len(face_encodings)} faces in the image.\n")
    return face_encodings, face_locations


def save_reference_image(image, face_location, output_dir, filename, resize_factor):
    print(f"Saving reference image: {filename}")
    top, right, bottom, left = face_location
    top = int(top / resize_factor)
    right = int(right / resize_factor)
    bottom = int(bottom / resize_factor)
    left = int(left / resize_factor)
    face_image = image[top:bottom, left:right]
    face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    face_path = os.path.join(output_dir, filename)
    cv2.imwrite(face_path, face_image_bgr)
    print(f"Reference image saved at: {face_path}\n")
    return face_path


def create_reference_images(known_faces, output_dir=REFERENCE_IMAGES_DIR, resize_factor=0.25):
    print("\nCreating reference images...")
    reference_encodings = []
    reference_images = []
    seen_faces = set()
    for path, encoding in known_faces:
        if encoding is None or any(
                face_recognition.compare_faces([ref_enc for ref_path, ref_enc in reference_encodings], encoding)):
            continue
        image = face_recognition.load_image_file(path)
        small_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
        face_locations = face_recognition.face_locations(small_image)
        for i, face_location in enumerate(face_locations):
            current_encoding = face_recognition.face_encodings(small_image, [face_location])[0]
            if face_recognition.compare_faces([current_encoding], encoding)[0]:
                reference_path = save_reference_image(image, face_location, output_dir,
                                                      f'reference_face_{len(reference_encodings)}.jpg', resize_factor)
                reference_encodings.append((reference_path, current_encoding))
                reference_images.append(reference_path)
                seen_faces.add(path)
                break
    print(f"Total reference images created: {len(reference_encodings)}\n")
    return reference_encodings, reference_images


def find_matching_faces(known_faces, selected_face_encoding, tolerance=0.55):
    print("\nFinding matching faces...")
    matches = []
    for path, encoding in known_faces:
        if encoding is None:
            continue
        match = face_recognition.compare_faces([encoding], selected_face_encoding, tolerance=tolerance)
        if match[0]:
            matches.append(path)
    print(f"Total matching faces found: {len(matches)}\n")
    return matches


def sort_photos_by_selected_face(known_faces, selected_face_index, resize_factor=0.25):
    print("\nSorting photos by selected face...")
    reference_faces, reference_images = create_reference_images(known_faces, resize_factor=resize_factor)
    print("-----------------------------------------------------")

    print("reference faces: ")
    print(reference_faces)
    print("-----------------------------------------------------")

    if len(reference_faces) <= selected_face_index:
        print(
            f"Error: Selected face index {selected_face_index} out of range. Only {len(reference_faces)} reference faces found.\n")
        return []

    selected_face_encoding = reference_faces[selected_face_index][1]
    matching_faces = find_matching_faces(known_faces, selected_face_encoding)
    return matching_faces


# CALL THIS WHEN THE USER UPLOADS A PHOTO

def identify_faces(input_directory):
    print("Loading and encoding faces from the directory...\n")

    # First check if there are new images in the input directory compared to the json file
    known_faces = load_encodings(ENCODINGS_FILE)
    processed_files = {path for path, _ in known_faces}
    new_files = []
    
    # Check if there are new images in the input directory that haven't been seen before
    for filename in os.listdir(input_directory):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(input_directory, filename)
                if path not in processed_files:
                    new_files.append(path)

    # Check if new_files is blank
    if not new_files:
        print("No new files found in the input directory.")
        return []
    # If new_files is not blank, process the new files only
    else:
        print(f"New files found in the input directory: {new_files}")
        # Load and encode faces from the directory once
        known_faces = load_and_encode_faces_from_image_paths(new_files)

        print("Creating reference images and presenting them for selection...\n")
        # Create reference images and present them to the user
        reference_encodings, reference_images = create_reference_images(known_faces)
        print("Reference faces available for selection:")
        for i, reference_image in enumerate(reference_images):
            print(f"{i}: {reference_image}")

        return reference_images











# CALL THIS WHEN THE USER SELECTS A FACE

def sort_by_face_index(input_directory, selected_face_index):
    # Simulate user selecting a face by index (for example, index 0)
    # selected_face_index = 15  # Replace with user input in a real application
    print(f"\nUser selected face index: {selected_face_index}\n")

    print("Sorting photos by the selected face...\n")
    known_faces = load_and_encode_faces_from_directory(input_directory)
    sorted_images = sort_photos_by_selected_face(known_faces, selected_face_index)
    print("Sorted images containing the selected face:")
    print(sorted_images)

if __name__ == '__main__':

    print(identify_faces(r'C:\Users\tdewa\Code\snapsort_project\SnapSort\static\uploads'))
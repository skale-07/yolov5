import os
import cv2
import face_recognition


def crop_faces_from_directory(input_directory, output_directory, resize_factor=0.25, margin=0.2):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    print(f"Processing images from directory: {input_directory}")
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_directory, filename)
            print(f"Processing file: {image_path}")
            image = face_recognition.load_image_file(image_path)
            small_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
            face_locations = face_recognition.face_locations(small_image)

            if face_locations:
                print(f"Found {len(face_locations)} faces in {image_path}")
                for i, face_location in enumerate(face_locations):
                    top, right, bottom, left = face_location
                    top = int(top / resize_factor)
                    right = int(right / resize_factor)
                    bottom = int(bottom / resize_factor)
                    left = int(left / resize_factor)

                    # Calculate margin size to expand the bounding box
                    height = bottom - top
                    width = right - left
                    margin_vertical = int(margin * height)
                    margin_horizontal = int(margin * width)

                    # Apply margins
                    top = max(0, top - margin_vertical)
                    right = min(image.shape[1], right + margin_horizontal)
                    bottom = min(image.shape[0], bottom + margin_vertical)
                    left = max(0, left - margin_horizontal)

                    face_image = image[top:bottom, left:right]
                    face_filename = f"{os.path.splitext(filename)[0]}_face_{i + 1}.jpg"
                    face_path = os.path.join(output_directory, face_filename)

                    # Convert the face image from RGB (face_recognition) to BGR (OpenCV) before saving
                    face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

                    # Save the cropped face image
                    success = cv2.imwrite(face_path, face_image_bgr)

                    if success:
                        print(f"Successfully saved cropped face to: {face_path}")
                    else:
                        print(f"Failed to save cropped face to: {face_path}")
            else:
                print(f"No faces found in {image_path}")
    print("Processing complete.")


# Example usage
input_directory = 'photos'
output_directory = 'cropped_output'
crop_faces_from_directory(input_directory, output_directory)

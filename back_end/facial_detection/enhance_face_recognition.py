import face_recognition
import os
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import matplotlib.pyplot as plt

def enhance_photo_with_pillow(image_path, output_path):
    # Open an image file
    with Image.open(image_path) as img:
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(4.0)

        # Save the enhanced image
        img.save(output_path)

def enhance_and_process_images(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            enhanced_image_path = os.path.join(output_folder, f"enhanced_{filename}")

            # Enhance the photo
            enhance_photo_with_pillow(image_path, enhanced_image_path)

            # Process the enhanced photo for face recognition and marking
            extract_and_mark_faces(enhanced_image_path, output_folder)

def extract_and_mark_faces(image_path, output_folder):

    # Load the enhanced image
    image = face_recognition.load_image_file(image_path)

    # Find all face locations and face encodings
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Convert to PIL format
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    for i, face_encoding in enumerate(face_encodings):
        top, right, bottom, left = face_locations[i]

        # Draw a rectangle around each face
        draw.rectangle([left, top, right, bottom], outline="red", width=2)

        # Draw the face number
        text_position = (left, top - 10)
        draw.text(text_position, str(i + 1), fill="red")

    # Save the annotated image
    annotated_image_path = os.path.join(output_folder, f"annotated_{os.path.basename(image_path)}")
    pil_image.save(annotated_image_path)

    print(f"Processed {image_path}, found {len(face_locations)} face(s).")

# Example usage
enhance_and_process_images("photos", "output_faces")

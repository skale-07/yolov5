import os
import time
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import json
import face_recognition
import cv2
import numpy as np
from collections import defaultdict
import datetime
import hashlib
# Importing necessary modules for sorting
from back_end.lighting_sort import lighting_sort_main as lighting_sort
from back_end.orientation.orientation_sort import sort_images_by_orientation as orientation_sort
from back_end.blur import blur_detection_with_object_detection as blur_sort
from back_end.object_detection.detect_objects import detect_objects
import back_end.facial_detection.facial_detection3 as facial_detection

# Flask app setup
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ENCODINGS_FILE = 'static/face_encodings.json'
REFERENCE_IMAGES_DIR = 'static/reference_images'
os.makedirs(REFERENCE_IMAGES_DIR, exist_ok=True)

# Helper functions for face recognition and sorting
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def remove_duplicate_images(folder_path):
    existing_hashes = set()
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                file_content = f.read()
                file_hash = hashlib.md5(file_content).hexdigest()

            if file_hash in existing_hashes:
                os.remove(file_path)
            else:
                existing_hashes.add(file_hash)

@app.route('/clear_selected', methods=['POST'])
def clear_selected():
    data = request.get_json()
    image_paths = data['imagePaths']
    print('session[uploaded_files]:', session['uploaded_files'])
    for path in image_paths:
        filename = path.split('/')[-1]
        file_path = os.path.join(app.static_folder, 'uploads', filename)
        try:
            os.remove(file_path)
            # remove from session['uploaded_files']
            print("filename",filename)
            session['uploaded_files'].remove(filename)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    return jsonify({"message": "Selected photos cleared."})

@app.route('/delete_all', methods=['POST'])
def delete_all_photos():
    # Delete all files in the upload folder
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    for file in files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Delete all files in the reference_images folder without deleting the folder itself
    reference_files = os.listdir(REFERENCE_IMAGES_DIR)
    for file in reference_files:
        file_path = os.path.join(REFERENCE_IMAGES_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Clear the face_encodings.json file
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'w') as file:
            json.dump([], file)  # Write an empty list to clear the file

    # Clear session data
    session.pop('uploaded_files', None)
    session.pop('sorted_images', None)
    session.pop('blurriness_scores', None)
    session.pop('reference_images', None)

    flash('All photos and reference images deleted.')
    return redirect(url_for('index'))

@app.route('/clear_all', methods=['POST'])
def clear_photos_from_workspace():
    session.pop('sorted_images', None)
    return redirect(url_for('index'))

@app.route('/clean_duplicates', methods=['POST'])
def clean_duplicates():
    try:
        remove_duplicate_images(REFERENCE_IMAGES_DIR)
        flash('Duplicate images removed successfully.')
    except Exception as e:
        flash(f'Error removing duplicate images: {e}')
        print(f'Error removing duplicate images: {e}')

    return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('file[]')
        all_detected_objects = []
        image_objects_dict = {}
        person_count_dict = {}
        blurriness_scores = {}

        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                session['uploaded_files'] = session.get('uploaded_files', []) + [filename]

                # Detect objects in the uploaded image (dummy function)
                detected_objects = detect_objects(file_path)
                detected_objects_list = list(detected_objects['name'])
                detected_bboxes = detected_objects[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name']]

                # Store the detected objects
                detected_objects_set = set(detected_objects_list)
                image_objects_dict[filename] = detected_objects_set
                all_detected_objects.append(detected_objects_set)

                # Count the number of persons in the image
                person_count = detected_objects_list.count('person')
                person_count_dict[filename] = person_count

                # Calculate blurriness score using the detected bounding boxes (dummy function)
                blurriness_score = blur_sort.calculate_blurriness(file_path, detected_bboxes)
                blurriness_scores[filename] = blurriness_score

        session['image_objects_dict'] = {k: list(v) for k, v in image_objects_dict.items()}
        session['person_count_dict'] = person_count_dict
        session['blurriness_scores'] = blurriness_scores

        if all_detected_objects:
            common_objects = set.union(*all_detected_objects)
            session['common_objects'] = list(common_objects)

        return redirect(url_for('index'))
    remove_duplicate_images(app.config['UPLOAD_FOLDER'])  # Call the function to remove duplicates
    return render_template('upload.html', common_objects=session.get('common_objects', []))

@app.route('/get_common_objects', methods=['GET'])
def get_common_objects():
    common_objects = session.get('common_objects', [])
    return jsonify({'common_objects': common_objects})

@app.route('/clear_objects', methods=['POST'])
def clear_objects():
    session['image_objects_dict'] = {}
    session['common_objects'] = []
    return jsonify({'message': 'Objects cleared from session'})

@app.route('/load_reference_images', methods=['GET'])
def load_reference_images():
    print("testing...")

    reference_images = facial_detection.identify_faces(app.config['UPLOAD_FOLDER'])

    # Edit the reference images so it only has static/reference_images/... paths
    reference_images = [os.path.join('static', 'reference_images', os.path.basename(image_path)) for image_path in reference_images]

    for img in reference_images:
        session["reference_images"] = session.get('reference_images', []) + [img]

    print("session['reference_images']: ", session["reference_images"])

    return jsonify({'reference_images': session["reference_images"]})

@app.route('/sort_by_selected_faces', methods=['POST'])
def sort_by_selected_faces():
    data = request.get_json()
    selected_faces = data.get('selectedFaces', [])
    if not selected_faces:
        return jsonify({'message': 'No faces selected', 'redirect': url_for('index')})

    known_faces = facial_detection.load_and_encode_faces_from_directory(app.config['UPLOAD_FOLDER'])
    sorted_images = []
    for face_filename in selected_faces:
        selected_face_index = int(face_filename.split('_')[-1].split('.')[0])
        temp = facial_detection.sort_photos_by_selected_face(known_faces, selected_face_index)
        print(temp)
        sorted_images.extend(temp)

    session['sorted_images'] = [os.path.basename(image_path) for image_path in sorted_images]
    print("Images sorted by selected faces: ", session["sorted_images"])

    return jsonify({'message': 'Photos sorted!', 'redirect': url_for('index')})

@app.route('/sort', methods=['POST'])
def sort_photos():
    lighting_checked = 'lighting' in request.form
    portrait_checked = 'portrait' in request.form
    landscape_checked = 'landscape' in request.form
    blur_checked = 'blurriness' in request.form
    object_value = request.form.get('object-dropdown')
    image_objects = session.get("image_objects_dict", {})
    person_count = request.form.get('person-count')
    person_count_checked = request.form.get('people-filter')
    blurriness_scores = session.get('blurriness_scores', {})

    facial_detection_checked = 'facial-detection' in request.form
    selected_faces = json.loads(request.form.get('selected-faces', '[]'))

    PHOTOS_DIR = 'static/uploads'
    uploaded_files = session.get('uploaded_files', [])
    sorted_images = []

    if facial_detection_checked and selected_faces:
        pass
        # Implement facial detection sorting logic here
        # known_faces = load_and_encode_faces_from_directory(PHOTOS_DIR)
        # selected_face_indices = [int(face.split('_')[-1].split('.')[0]) for face in selected_faces]
        # sorted_images = []
        # for index in selected_face_indices:
        #     sorted_images.extend(sort_photos_by_selected_face(known_faces, index))
        # session['sorted_images'] = [os.path.basename(image_path) for image_path in sorted_images]

    elif lighting_checked and (portrait_checked or landscape_checked):
        if portrait_checked:
            sorted_images = orientation_sort(PHOTOS_DIR, 'portrait')
        if landscape_checked:
            sorted_images = orientation_sort(PHOTOS_DIR, 'landscape')
        sorted_images = lighting_sort.process_images_parallel_image_paths(sorted_images)
        session['sorted_images'] = [os.path.basename(image_path) for image_path, _ in sorted_images]
    elif lighting_checked:
        start_time = time.time()
        sorted_images = lighting_sort.process_images_parallel_directory(PHOTOS_DIR)
        session['sorted_images'] = [os.path.basename(image_path) for image_path, _ in sorted_images]
    elif portrait_checked or landscape_checked:
        if portrait_checked:
            sorted_images = orientation_sort(PHOTOS_DIR, 'portrait')
            session['sorted_images'] = [os.path.basename(image_path) for image_path in sorted_images]
        if landscape_checked:
            sorted_images = orientation_sort(PHOTOS_DIR, 'landscape')
            session['sorted_images'] = [os.path.basename(image_path) for image_path in sorted_images]
    elif person_count_checked:
        sorted_images = [img for img in uploaded_files if session['person_count_dict'].get(img, 0) >= int(person_count)]
        session['sorted_images'] = [os.path.basename(image_path) for image_path in sorted_images]
    elif object_value:
        sorted_images = [img for img in uploaded_files if object_value in image_objects.get(img, [])]
        session['sorted_images'] = [os.path.basename(image_path) for image_path in sorted_images]
    elif blur_checked:
        sorted_images = sorted(blurriness_scores.items(), key=lambda item: item[1], reverse=True)
        session['sorted_images'] = [os.path.basename(image_path) for image_path, _ in sorted_images]

    if not sorted_images:
        session['sorted_images'] = sorted_images

    flash('Photos sorted! (placeholder message)')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get('PORT', 8080)))

from flask import Flask, request, jsonify, stream_with_context
import face_recognition
import cv2
import os
import time
import json

app = Flask(__name__)

def locate_person_in_folder(target_image_path, folder_path, tolerance=0.6):
    # Load target image
    target_image = face_recognition.load_image_file(target_image_path)
    target_locations = face_recognition.face_locations(target_image)
    target_encoding = face_recognition.face_encodings(target_image, target_locations)[0]

    results = []

    # Get the total number of images for progress estimation
    total_images = sum(1 for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png')))

    # Iterate through all files in the folder
    for i, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Adjust the file extensions as needed
            image_path = os.path.join(folder_path, filename)

            # Load group image and detect faces
            group_image = cv2.imread(image_path)
            group_locations = face_recognition.face_locations(group_image)

            for j, face_location in enumerate(group_locations):
                face_encoding = face_recognition.face_encodings(group_image, [face_location])[0]

                # Compare face encoding with target face encoding using tolerance
                match = face_recognition.compare_faces([target_encoding], face_encoding, tolerance=tolerance)

                # Add result to the list
                if match[0]:
                    results.append({"image_path": image_path})

            # Calculate progress as a percentage
            progress = int((i + 1) / total_images * 100)

            # Yield the progress
            yield (json.dumps({"status": "loading", "progress": progress, "message": "Processing request..."}),)

            # Introduce a small delay for demonstration purposes
            time.sleep(0.1)

    # Yield the final results and progress
    yield (json.dumps({"status": "success", "progress": 100, "message": "Request processed successfully", "results": results}),)

@app.route('/locate_person', methods=['GET'])
def locate_person():
    # Get parameters from query string
    target_image_path = request.args.get('target_image_path')
    group_images_folder = request.args.get('group_images_folder')
    tolerance = float(request.args.get('tolerance', 0.6))

    # Validate parameters
    if not (target_image_path and group_images_folder):
        return jsonify({"error": "Please provide target_image_path and group_images_folder"}), 400

    # Stream the response with context
    return stream_with_context(locate_person_in_folder(target_image_path, group_images_folder, tolerance=tolerance))

if __name__ == '__main__':
    app.run(debug=True)




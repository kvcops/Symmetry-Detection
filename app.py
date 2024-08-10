from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import tempfile
import os
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)

def load_image(file):
    """
    Save and load the uploaded image.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        temp_file_path = temp_file.name
        file.save(temp_file_path)
    
    img = cv2.imread(temp_file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Unable to load image")
    
    os.remove(temp_file_path)
    return img

def preprocess_image(img):
    """
    Threshold and detect edges.
    """
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary, 50, 150)
    return edges

def check_symmetry(img, axis_type, threshold=0.95):
    """
    Check symmetry along a specified axis.
    """
    h, w = img.shape[:2]
    if axis_type == "vertical":
        mid = w // 2
        left = img[:, :mid]
        right = cv2.flip(img[:, -mid:], 1)
        similarity = np.sum(left == right) / (h * mid)
    elif axis_type == "horizontal":
        mid = h // 2
        top = img[:mid, :]
        bottom = cv2.flip(img[-mid:, :], 0)
        similarity = np.sum(top == bottom) / (mid * w)
    elif axis_type == "diagonal":
        flipped = cv2.flip(img, -1)
        similarity = np.sum(img == flipped) / (h * w)
    return similarity > threshold

def check_rotational_symmetry(img, n=3, threshold=0.95):
    """
    Check for n-fold rotational symmetry.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    angle = 360 // n
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (w, h))
    similarity = np.sum(img == rotated) / (h * w)
    return similarity > threshold

def draw_symmetry_line(img, axis_type):
    """
    Draw symmetry lines on the image.
    """
    h, w = img.shape[:2]
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if axis_type == "vertical":
        cv2.line(result, (w//2, 0), (w//2, h), (0, 255, 0), 2)
    elif axis_type == "horizontal":
        cv2.line(result, (0, h//2), (w, h//2), (0, 255, 0), 2)
    elif axis_type == "diagonal":
        cv2.line(result, (0, 0), (w, h), (0, 255, 0), 2)
        cv2.line(result, (w, 0), (0, h), (0, 255, 0), 2)
    return result

def draw_rotational_symmetry(img, n):
    """
    Draw lines for rotational symmetry.
    """
    h, w = img.shape[:2]
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    center = (w // 2, h // 2)
    radius = min(h, w) // 4
    for i in range(n):
        angle = i * (360 / n)
        x = int(center[0] + radius * np.cos(np.radians(angle)))
        y = int(center[1] + radius * np.sin(np.radians(angle)))
        cv2.line(result, center, (x, y), (0, 255, 0), 2)
    return result

def analyze_symmetry(img):
    """
    Analyze and return symmetry results.
    """
    preprocessed = preprocess_image(img)
    results = []
    for axis in ["vertical", "horizontal", "diagonal"]:
        if check_symmetry(preprocessed, axis):
            img_with_line = draw_symmetry_line(img, axis)
            results.append((f"{axis.capitalize()} symmetry", img_with_line))
    
    for n in [2, 3, 4]:  # Check 2, 3, and 4-fold symmetry
        if check_rotational_symmetry(preprocessed, n):
            img_with_rotation = draw_rotational_symmetry(img, n)
            results.append((f"{n}-fold rotational symmetry", img_with_rotation))
    
    if not results:
        results.append(("No symmetry detected", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    
    return results

def encode_image(img):
    """
    Encode image as base64 string.
    """
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    """
    Render the home page.
    """
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Handle image upload and analysis.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    
    try:
        img = load_image(file)
    except ValueError:
        return jsonify({'error': 'Invalid image'}), 400

    results = analyze_symmetry(img)
    encoded_results = [{'title': title, 'image': encode_image(img)} for title, img in results]

    return jsonify(encoded_results)

if __name__ == '__main__':
    app.run(debug=True)

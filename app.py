from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import cv2
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads'
CLOTHES_FOLDER = './static/images/clothes'
JEWELRY_FOLDER = './static/images/jewelry'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CLOTHES_FOLDER'] = CLOTHES_FOLDER
app.config['JEWELRY_FOLDER'] = JEWELRY_FOLDER

# Load the trained Random Forest model
model_path = './model/skin_tone_classifier.pkl'
with open(model_path, 'rb') as model_file:
    clf = pickle.load(model_file)

# Mapping of seasons to jewelry types
JEWELRY_MAP = {
    'light spring': ['gold.png', 'copper.png'],
    'bright spring': ['gold.png', 'white-gold.png'],
    'warm spring': ['gold.png', 'copper.png'],
    'light summer': ['white-gold.png', 'rose-gold.png'],
    'muted summer': ['white-gold.png', 'rose-gold.png'],
    'cool summer': ['white-gold.png'],
    'dark autumn': ['gold.png', 'copper.png'],
    'muted autumn': ['gold.png', 'copper.png'],
    'warm autumn': ['gold.png', 'copper.png'],
    'dark winter': ['white-gold.png'],
    'bright winter': ['white-gold.png'],
    'cool winter': ['white-gold.png'],
}

# Helper functions

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def crop_to_square(image_path):
    img = Image.open(image_path)
    width, height = img.size

    if width == height:
        return image_path

    # Crop the center square
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2

    img_cropped = img.crop((left, top, right, bottom))
    cropped_image_path = image_path  # Overwrite the original image
    img_cropped.save(cropped_image_path)
    
    return cropped_image_path

def apply_grabcut(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        rect = (x, y, w, h)
    else:
        return None, None  # Return None if no face is detected

    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image = image * mask2[:, :, np.newaxis]

    return image, faces

def get_dominant_colors(image, k=3):
    if image is None:
        return []  # Return empty list if no image is passed
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Sort colors by frequency to ensure consistency
    unique, counts = np.unique(labels, return_counts=True)
    dominant_colors = [tuple(centers[i].astype(np.uint8)) for i in unique[np.argsort(-counts)]]

    return dominant_colors  # Return the list as is, even if it contains fewer than 3 colors

def filter_black_colors(colors):
    return [color for color in colors if not all(component <= 1 for component in color)]

def calculate_centroid(colors):
    if not colors:
        return (0, 0, 0)  # Return black if no colors are provided
    colors_array = np.array(colors, dtype=np.float32)
    centroid = np.mean(colors_array, axis=0)
    return tuple(centroid.astype(np.uint8))

def convert_colors(color):
    rgb = np.array(color, dtype=np.uint8)
    rgb_for_conversion = rgb.reshape(1, 1, 3)
    hsv = cv2.cvtColor(rgb_for_conversion, cv2.COLOR_RGB2HSV).reshape(3)
    ycbcr = cv2.cvtColor(rgb_for_conversion, cv2.COLOR_RGB2YCrCb).reshape(3)
    return np.concatenate((rgb, hsv, ycbcr))

def process_image(image_path, k=3):
    # Crop the image to a square
    cropped_image_path = crop_to_square(image_path)
    
    # Continue with the existing image processing (e.g., GrabCut, etc.)
    image = cv2.imread(cropped_image_path)
    processed_image, faces = apply_grabcut(image)
    
    # If no face detected, return early
    if processed_image is None:
        return None, None, None, False
    
    dominant_colors = get_dominant_colors(processed_image, k)
    dominant_colors = filter_black_colors(dominant_colors)
    centroid = calculate_centroid(dominant_colors)
    features = convert_colors(centroid)
    
    return features, dominant_colors, centroid, True  # True indicates that a face was detected


@app.route('/', methods=['GET', 'POST'])
def index():
    no_face_detected = False  # Set default value for flag
    if request.method == 'POST':
        if 'file' not in request.files or 'gender' not in request.form:
            return redirect(request.url)
        file = request.files['file']
        gender = request.form['gender']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image
            features, dominant_colors, centroid, face_detected = process_image(filepath)
            if not face_detected:
                # Return to the index with a flag indicating no face was detected
                return render_template('index.html', no_face_detected=True)

            features = features.reshape(1, -1)

            # Predict the season using the Random Forest model
            predicted_season = clf.predict(features)[0]
            season_folder = predicted_season.lower().replace(' ', '_')

            # Determine the folder based on gender and season
            clothes_folder = os.path.join(app.config['CLOTHES_FOLDER'], gender, season_folder)
            if not os.path.exists(clothes_folder):
                recommended_images = []
            else:
                recommended_images = os.listdir(clothes_folder)

            # Get the correct jewelry images based on the season
            jewelry_images = JEWELRY_MAP.get(predicted_season.lower(), [])

            # Determine the palette image path
            palette_image = f'palette/{season_folder}.png'

            return render_template(
                'result.html', 
                season=predicted_season, 
                dominant_colors=dominant_colors, 
                centroid=centroid, 
                filename=filename, 
                season_folder=season_folder,
                recommended_images=recommended_images,
                jewelry_images=jewelry_images,
                gender=gender,
                palette_image=palette_image  # Pass the palette image path to the template
            )

    return render_template('index.html', no_face_detected=no_face_detected)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# Route for the homepage
@app.route('/home')
def home():
    return redirect(url_for('index'))  # Redirect to index route

# Route for the About page
@app.route('/about')
def about():
    return render_template('About.html')

# Route for the Contact page
@app.route('/contact')
def contact():
    return render_template('Contact.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Mengambil port dari environment variable, default ke 5000
    app.run(host='0.0.0.0', port=port, debug=True)  # Set host ke 0.0.0.0 untuk menerima koneksi eksternal


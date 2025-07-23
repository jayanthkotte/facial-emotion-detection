from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import torch.nn as nn
from torchvision.models import resnet50
import cv2
import numpy as np

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pt')

# Define emotion classes
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load OpenCV Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_model(num_classes):
    model = resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model

def detect_and_crop_face(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        raise ValueError("No face detected in the image")

    # Crop the first detected face
    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]
    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    return face_pil

def prepare_image(image_bytes):
    face_image = detect_and_crop_face(image_bytes)
    image = preprocess(face_image)
    image = image.unsqueeze(0)
    return image

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    img_bytes = file.read()
    try:
        input_tensor = prepare_image(img_bytes)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            emotion = EMOTION_CLASSES[predicted.item()]
            confidence_score = confidence.item()
        return jsonify({
            'emotion': emotion,
            'confidence': round(confidence_score * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    NUM_CLASSES = 7
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    app.run(host='0.0.0.0', port=5000, debug=True)

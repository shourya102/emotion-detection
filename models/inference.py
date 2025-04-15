import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.train import MultiModalEmotic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_DIR = os.path.dirname(__file__)

context_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

body_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

thresholds = np.load(os.path.join(CURRENT_DIR, "thresholds.npy"))


def infer_emotion(image_path, model, device, thresholds, ind2cat, ind2vad, face_detector):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    context_img = context_transform(image).unsqueeze(0).to(device)
    cv_image = cv2.imread(image_path)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        print("No face detected! Using full image as body.")
        body_img = context_img
    else:
        x, y, w, h = faces[0]
        body_crop = cv_image[y:y + h, x:x + w]
        body_crop = cv2.cvtColor(body_crop, cv2.COLOR_BGR2RGB)
        body_img = Image.fromarray(body_crop)
        body_img = body_transform(body_img).unsqueeze(0).to(device)
    with torch.no_grad():
        cont_pred, cat_pred = model(context_img, body_img)
    cont_pred = cont_pred.cpu().numpy().flatten()
    cat_pred = cat_pred.cpu().numpy().flatten()
    predicted_emotions = [ind2cat[i] for i in range(len(cat_pred)) if cat_pred[i] > thresholds[i]]
    cont_emotions = {ind2vad[i]: cont_pred[i].item() for i in range(3)}
    return predicted_emotions, cont_emotions


def get_infer(image_path):
    model = MultiModalEmotic().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(CURRENT_DIR, "best_emotic_model.pth"), map_location=DEVICE))
    model.eval()
    ind2cat = {i: cat for i, cat in enumerate(['Peace', 'Affection', 'Esteem', 'Anticipation', 'Engagement',
                                               'Confidence', 'Happiness', 'Pleasure', 'Excitement', 'Surprise',
                                               'Sympathy', 'Doubt/Confusion', 'Disconnection', 'Fatigue',
                                               'Embarrassment', 'Yearning', 'Disapproval', 'Aversion', 'Annoyance',
                                               'Anger', 'Sensitivity', 'Sadness', 'Disquietment', 'Fear', 'Pain',
                                               'Suffering'])}
    ind2vad = {0: 'Valence', 1: 'Arousal', 2: 'Dominance'}
    face_cascade = cv2.CascadeClassifier(os.path.join(CURRENT_DIR, "haarcascade_frontalface_default.xml"))
    predicted_emotions, cont_values = infer_emotion(image_path, model, DEVICE, thresholds, ind2cat, ind2vad,
                                                    face_cascade)
    return predicted_emotions, cont_values

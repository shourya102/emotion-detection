import base64
import io
import os
from datetime import datetime

from PIL import Image
from flask import request, jsonify, Blueprint, render_template

from models.inference import get_infer

base = Blueprint('routes', __name__)
CURRENT_DIR = os.path.dirname(__file__)
SAVE_DIR = os.path.join(CURRENT_DIR, "upload")
os.makedirs(SAVE_DIR, exist_ok=True)


@base.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    data = request.json
    image_data = data.get('image', '')
    if not image_data:
        return render_template('index.html', error='No image data found')
    try:
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = os.path.join(SAVE_DIR, filename)
        image.save(image_path, "JPEG")
        data = get_infer(image_path)
        return jsonify({
            'emotion': ', '.join(data[0]),
            'vad': {
                'Valence': round(data[1]['Valence'], 2),
                'Arousal': round(data[1]['Arousal'], 2),
                'Dominance': round(data[1]['Dominance'], 2)
            }
        })
    except Exception as e:
        return render_template('index.html', error=str(e))


@base.route('/')
def index():
    return render_template('index.html')

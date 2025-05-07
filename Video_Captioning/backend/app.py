from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import torch
from transformers import AutoProcessor
import cv2
from PIL import Image

app = Flask(__name__, static_folder='../frontend')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
processor = AutoProcessor.from_pretrained("microsoft/git-base")

# CORS support
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("models/model.pt", weights_only=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_frames(video_path, num_frames=40):
    """Extract frames using OpenCV"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(num_frames):
        frame_idx = int(i * (total_frames / num_frames))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    return frames


def process_video(video_path):
    """Process video and generate caption"""
    try:
        images = extract_frames(video_path)
        captions = []

        for image in images:
            inputs = processor(images=[image], return_tensors="pt").to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=45)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            captions.append(caption)

        return captions[0]

    except Exception as e:
        print(f"Error processing video: {e}")
        return None


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate_caption', methods=['POST'])
def generate_caption():
    try:
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'error': 'Filename not provided'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], data['filename'])
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404

        caption = process_video(filepath)
        if not caption:
            return jsonify({'error': 'Failed to generate caption'}), 500

        # Clean up the uploaded file
        os.remove(filepath)
        return jsonify({'caption': caption}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5001, debug=True)
import cv2
import torch
from torchvision import transforms, models
import numpy as np
import tempfile
from collections import Counter
from moviepy.editor import VideoFileClip
import os
import librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# Load facial model (ResNet-50)
facial_model = models.resnet50(weights=None)
num_ftrs = facial_model.fc.in_features
facial_model.fc = torch.nn.Linear(num_ftrs, 7)
facial_model.load_state_dict(torch.load('../models/resnet50_fer.pth', map_location=device))
facial_model = facial_model.to(device)
facial_model.eval()

class SpeechCNN(torch.nn.Module):
    def __init__(self):
        super(SpeechCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 3 * 32, 128)
        self.fc2 = torch.nn.Linear(128, 7)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

speech_model = SpeechCNN().to(device)
speech_model.load_state_dict(torch.load('../models/cnn_tess.pth', map_location=device))
speech_model.eval()

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
facial_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_facial_feedback(emotion, confidence):
    return f"{emotion} ({confidence:.2f}%)"

def get_speech_feedback(emotion, confidence):
    return f"{emotion} ({confidence:.2f}%)"

def process_facial_emotions(video_path, sample_rate=1):  # Process every frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Get total frame count for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    
    # Try multiple codecs to ensure compatibility
    codecs = [cv2.VideoWriter_fourcc(*'XVID'), cv2.VideoWriter_fourcc(*'MJPG')]
    out = None
    for codec in codecs:
        out = cv2.VideoWriter(temp_file.name, codec, fps, (width, height))
        if out.isOpened():
            break
    if out is None:
        raise ValueError("Could not initialize video writer with available codecs")

    facial_frame_feedback = []
    facial_emotion_counts = Counter()
    total_faces = 0
    frame_count = 0
    emotion_timeline = []

    # Batch processing for better GPU utilization
    batch_size = 16  # Process frames in batches for efficiency
    batch_frames = []
    batch_emotions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:  # Now processes every frame (sample_rate=1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            frame_emotions = []

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_tensor = facial_transform(face_rgb).unsqueeze(0).to(device)
                batch_frames.append(face_tensor)
                batch_emotions.append((frame_count, frame, x, y, w, h))  # Store frame info for later

            processed_frames += 1

        frame_count += 1

        # Process batch when full or at end
        if len(batch_frames) >= batch_size or (not ret and batch_frames):
            if batch_frames:
                batch_tensors = torch.cat(batch_frames, dim=0).to(device)
                with torch.no_grad():
                    outputs = facial_model(batch_tensors)
                    probs = torch.softmax(outputs, dim=1)
                    confidences, predictions = torch.max(probs, 1)

                for i, (frame_count, frame, x, y, w, h) in enumerate(batch_emotions):
                    emotion = emotions[predictions[i].item()]
                    confidence_percent = confidences[i].item() * 100

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    text = f"{emotion} ({confidence_percent:.2f}%)"
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    frame_emotions.append(get_facial_feedback(emotion, confidence_percent))
                    facial_emotion_counts[emotion] += 1
                    total_faces += 1
                    emotion_timeline.append((frame_count, emotion, confidence_percent))

                facial_frame_feedback.append(f"Frame {frame_count + 1}: {', '.join(frame_emotions) if frame_emotions else 'No faces detected'}")
                out.write(frame)
                print(f"Wrote frame {frame_count}")  # Debug: Track frame writing
                batch_frames.clear()
                batch_emotions.clear()

    cap.release()
    out.release()

    # Return processed frames count for progress
    facial_overall = [f"{emotion}: {(facial_emotion_counts[emotion] / total_faces) * 100:.2f}%" for emotion in emotions] if total_faces > 0 else ["No faces detected"]
    return temp_file.name, facial_frame_feedback, facial_overall, facial_emotion_counts, total_faces, emotion_timeline, processed_frames

def process_speech_emotions(audio_path, chunk_length=2.0):  # Updated to accept .wav path
    try:
        print(f"Loading audio from {audio_path}")  # Debug log

        audio, sr = librosa.load(audio_path, sr=22050)
        print(f"Audio length: {len(audio)} samples, Sample rate: {sr}")  # Debug log

        chunk_samples = int(chunk_length * sr)
        max_len = 128
        speech_emotion_counts = Counter()
        total_chunks = 0
        speech_frame_feedback = []
        emotion_timeline = []
        processed_chunks = 0
        total_possible_chunks = len(audio) // chunk_samples

        print(f"Processing {total_possible_chunks} chunks of {chunk_length} seconds each")  # Debug log

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) < chunk_samples // 2:
                print(f"Skipping short chunk at index {i} (length {len(chunk)})")  # Debug log
                continue
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
            if mfcc.shape[1] < max_len:
                mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
            else:
                mfcc = mfcc[:, :max_len]
            mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    outputs = speech_model(mfcc_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                    emotion = emotions[predicted.item()]
                    confidence_percent = confidence.item() * 100
            except Exception as model_error:
                print(f"Error processing chunk at index {i}: {str(model_error)}")  # Debug log
                continue

            speech_frame_feedback.append(f"Chunk {total_chunks + 1} ({i/sr:.1f}-{min((i + chunk_samples)/sr, len(audio)/sr):.1f}s): {get_speech_feedback(emotion, confidence_percent)}")
            speech_emotion_counts[emotion] += 1
            total_chunks += 1
            emotion_timeline.append((i/sr, emotion, confidence_percent))
            processed_chunks += 1
            print(f"Processed chunk {processed_chunks} of {total_possible_chunks}")  # Debug log

        if total_chunks == 0:
            raise ValueError("No valid audio chunks processed")

        speech_overall = [f"{emotion}: {(speech_emotion_counts[emotion] / total_chunks) * 100:.2f}%" for emotion in emotions] if total_chunks > 0 else ["No audio detected"]
        
        # Return processed chunks count for progress
        return speech_frame_feedback, speech_overall, speech_emotion_counts, total_chunks, emotion_timeline, processed_chunks

    except Exception as e:
        print(f"Error in process_speech_emotions: {str(e)}")  # Debug log
        raise

def process_live_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    emotions_detected = []

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_tensor = facial_transform(face_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = facial_model(face_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            emotion = emotions[predicted.item()]
            confidence_percent = confidence.item() * 100

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{emotion} ({confidence_percent:.2f}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        emotions_detected.append((emotion, confidence_percent))

    return frame, emotions_detected
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load a pre-trained CNN model (ResNet) for feature extraction
class VideoFeatureExtractor:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove last FC layer
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, frames):
        features = []
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame).unsqueeze(0)
            with torch.no_grad():
                feature = self.model(frame).squeeze().numpy()
            features.append(feature)
        return np.array(features)

# LSTM-based sequence model for summarization
class VideoSummarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(VideoSummarizer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 512)  # Feature vector output

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        summary_features = self.fc(lstm_out[:, -1, :])  # Last LSTM output
        return summary_features

# Load a pre-trained text summarization model
class TextGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    def generate_summary(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
        summary_ids = self.model.generate(inputs["input_ids"], max_length=100, min_length=30, length_penalty=2.0)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Load Video and Extract Frames
def load_video(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps // frame_rate if fps >= frame_rate else 1
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

# Main function to generate summary from a video
def generate_video_summary(video_path):
    feature_extractor = VideoFeatureExtractor()
    summarizer = VideoSummarizer(input_size=2048, hidden_size=512, num_layers=2)
    text_generator = TextGenerator()

    # Extract video frames
    frames = load_video(video_path)
    if not frames:
        return "No frames extracted from video."

    # Extract features from frames
    features = feature_extractor.extract_features(frames)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    # Generate feature-based summary
    with torch.no_grad():
        summary_features = summarizer(features_tensor).numpy()

    # Convert features to a text format (naive approach)
    feature_text = " ".join([f"Scene with feature vector {i}" for i in summary_features.flatten()[:50]])

    # Generate textual summary
    summary = text_generator.generate_summary(feature_text)
    return summary

# Example usage
video_path = "sample_video.mp4"  # Replace with actual video path
summary = generate_video_summary(video_path)
print("Video Summary:", summary)
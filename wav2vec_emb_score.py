import torch
from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import numpy as np
import librosa

# Load datasets
data_files = {
    "train": "./train_data2.csv",
    "validation": "./dev_data.csv",
    "test": "./test_data.csv"
}
dataset = load_dataset("csv", data_files=data_files, delimiter=",")

# Specify the input and output columns
input_column = "file_path"
output_column = "Emotion"

# Load the saved feature extractor and model
model_path = "./wav2vec2-emotion-recognition"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
model = AutoModelForAudioClassification.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define function to load and preprocess audio
def preprocess_function(examples):
    max_length = 16000 * 10  # 10 seconds at 16kHz sampling rate
    
    def load_audio(file_path):
        audio, sr = librosa.load(file_path, sr=16000)
        if len(audio) > max_length:
            audio = audio[:max_length]
        elif len(audio) < max_length:
            padding = np.zeros(max_length - len(audio))
            audio = np.concatenate((audio, padding))
        return audio

    audio_arrays = [load_audio(file_path) for file_path in examples[input_column]]
    
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    return inputs

# Function to extract scores and labels
def extract_scores(dataset):
    scores = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            inputs = preprocess_function(dataset[i:i+1])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass to get logits
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()  # Get probabilities
            
            scores.append(probs)
            labels.append(dataset[i][output_column])
    
    return np.vstack(scores), np.array(labels)

# Extract and save scores for each dataset
for split in ["train", "validation", "test"]:
    print(f"Processing {split} dataset...")
    
    scores, labels = extract_scores(dataset[split])
    
    # Save scores
    np.save(f"{split}_scores.npy", scores)
    
    # Save labels as .npy
    np.save(f"{split}_labels.npy", labels)
    
    print(f"Saved {split}_scores.npy and {split}_labels.npy")

print("Score extraction complete!")


import torch
from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import numpy as np
import librosa
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

# Load datasets
data_files = {
    "train": "./train_data2.csv",
    "validation": "./dev_data.csv",
    "test": "./test_data.csv"
}
dataset = load_dataset("csv", data_files=data_files, delimiter=",")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
test_dataset = dataset["test"]

# Specify the input and output columns
input_column = "file_path"
output_column = "Emotion"

# Get unique labels
label_list = train_dataset.unique(output_column)
label_list.sort()
num_labels = len(label_list)
print(f"A classification problem with {num_labels} classes: {label_list}")

# Create label to id mapping
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# Load pre-trained model and feature extractor
model_name = "r-f/wav2vec-english-speech-emotion-recognition"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Define Voice Activity Detection (VAD) function
def voice_activity_detection(audio, threshold=0.01, frame_length=1024, hop_length=512):
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    voiced_indices = np.where(energy > threshold)[0]
    
    if len(voiced_indices) == 0:
        return audio  # Return original audio if no voice activity detected
    
    start_sample = voiced_indices[0] * hop_length
    end_sample = (voiced_indices[-1] + 1) * hop_length
    
    voiced_audio = audio[start_sample:end_sample]
    return voiced_audio
# Define SpecAugmentation class
class SpecAugment(nn.Module):
    def __init__(self, rate, policy=3, freq_mask=15, time_mask=35):
        super(SpecAugment, self).__init__()
        self.rate = rate
        self.policy = policy
        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def forward(self, x):
        if self.training:
            b, c, t, f = x.size()
            mask = torch.ones((b, t, f), device=x.device)
            for idx in range(b):
                for _ in range(self.policy):
                    if torch.rand(1) < self.rate:
                        t_start = int(torch.rand(1) * (t - self.time_mask))
                        t_end = t_start + self.time_mask
                        mask[idx, t_start:t_end, :] = 0
                    if torch.rand(1) < self.rate:
                        f_start = int(torch.rand(1) * (f - self.freq_mask))
                        f_end = f_start + self.freq_mask
                        mask[idx, :, f_start:f_end] = 0
            x = x * mask.unsqueeze(1)
        return x

# Custom model with SpecAugmentation
class Wav2Vec2ForSpeechClassification(nn.Module):
    def __init__(self, num_labels, model_name):
        super().__init__()
        self.num_labels = num_labels
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, num_labels)
        self.spec_augment = SpecAugment(rate=0.5)

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        hidden_states = self.spec_augment(hidden_states.unsqueeze(1)).squeeze(1)
        logits = self.classifier(hidden_states[:, 0, :])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {'loss': loss, 'logits': logits}

# Initialize the custom model
model = Wav2Vec2ForSpeechClassification(num_labels=num_labels, model_name=model_name)

# Define function to load and preprocess audio
def preprocess_function(examples):
    max_length = 16000 * 10  # 10 seconds at 16kHz sampling rate
    
    def load_audio(file_path):
        audio, sr = librosa.load(file_path, sr=16000)
        audio = voice_activity_detection(audio)  # Apply VAD
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
    inputs["labels"] = [label2id[label] for label in examples[output_column]]
    return inputs
# Preprocess the datasets
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)
test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    load_best_model_at_end=True,
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=lambda p: {
        "accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()
    },
)

# Train the model
trainer.train()

# Evaluate the model on the test set
test_results = trainer.evaluate(test_dataset)
print(f"Test results: {test_results}")

# Save the model
model.save_pretrained("./wav2vec2-emotion-recognitionaug")
feature_extractor.save_pretrained("./wav2vec2-emotion-recognitionaug")

# Save the feature extractor part of the model
feature_extractor_model = model.wav2vec2
feature_extractor_model.save_pretrained("./wav2vec2-emotion-recognition-feature-extractoraug")

import torch
from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import numpy as np
import librosa

# Load datasets
data_files = {
    "train": "./train_voice_sent.csv",
    "validation": "./dev_voice_sent.csv",
    "test": "./test_voice_sent.csv"  # Add the test dataset
}
dataset = load_dataset("csv", data_files=data_files, delimiter=",")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
test_dataset = dataset["test"]  # Load the test dataset

# Specify the input and output columns
input_column = "file_path"
output_column = "Sentiment"

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
model = AutoModelForAudioClassification.from_pretrained(
    model_name, 
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label
)

# Define function to load and preprocess audio
def preprocess_function(examples):
    max_length = 16000 * 10  # 10 seconds at 16kHz sampling rate
    
    def load_audio(file_path):
        audio, sr = librosa.load(file_path, sr=16000)
        if len(audio) > max_length:
            audio = audio[:max_length]
        elif len(audio) < max_length:
            # Pad audio if it's shorter than 10 seconds
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
    num_train_epochs=10,
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
model.save_pretrained("./wav2vec2-sent-recognition")
feature_extractor.save_pretrained("./wav2vec2-sent-recognition")

# Save the feature extractor part of the model
feature_extractor_model = model.wav2vec2
feature_extractor_model.save_pretrained("./wav2vec_sent-recognition-feature-extractor")




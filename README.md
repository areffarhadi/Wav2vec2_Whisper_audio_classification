# Wav2vec2 audio classification
Using this script, we can implement several scenarios in audio classification, such as speaker identification, language recognition, emotion recognition, sentiment analysis and more, using Wav2vec2 and Whisper models. 

For fine-tuning the Whisper model for audio classification: [Whisper_Emotion.py](https://github.com/areffarhadi/Wav2vec2_audio_classification/blob/main/Whisper_Emotion.py) <be>

For fine-tuning Wav2Vec2 for audio classification: [wav2vec_Emotion.py](https://github.com/areffarhadi/Wav2vec2_audio_classification/blob/main/wav2vec_Emotion.py)

The manifest for feeding wav data must be like [train_voice_emotion.csv](https://github.com/areffarhadi/Wav2vec2_audio_classification/blob/main/train_voice_emotion.csv) file.

in addition we have [wav2vec_Emotion_specaugm.py](https://github.com/areffarhadi/Wav2vec2_audio_classification/blob/main/wav2vec_Emotion_specaugm.py) for utilizing SpecAugment as augmentation technique, [wav2vec_embeding.py](https://github.com/areffarhadi/Wav2vec2_audio_classification/blob/main/wav2vec_embeding.py) for extract and save feature embedings and [wav2vec_emb_score.py](https://github.com/areffarhadi/Wav2vec2_audio_classification/blob/main/wav2vec_emb_score.py) for extracting scores for each wav file.

Please use [slurm_run.sh](https://github.com/areffarhadi/Wav2vec2_audio_classification/blob/main/slurm_run.sh) to run the scripts using Slurm.




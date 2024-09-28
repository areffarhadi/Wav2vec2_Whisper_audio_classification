#!/bin/bash
### Here are the SBATCH parameters that you should always consider:

### Here are the SBATCH parameters that you should always consider:
#SBATCH --gpus=A100:1
# SBATCH --gpus=1
#SBATCH --time=0-10:00:00   ## days-hours:minutes:seconds
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=20
# SBATCH --output=fine_tuning.out

# Check if the folder path is provided as an argument

# source /home/arfarh/scratch/Aref_tools/whisper-venv/bin/activate
# Directory containing the .wav files

module load gpu
module load cudnn
module load cuda/11.8.0
source /home/arfarh/scratch/Aref_tools/whisper-venv/bin/activate
# python3 ./wav2vec_Emotion_specaugm.py
# python3 ./wav2vec_embeding_claud.py
# python3 ./wav2vec_Emotion.py
python3 ./Whisper_Emotion.py
# python3 ./code3.py
# python3 ./test_checkpoint.py
# python3 ./Aref_whisp_finetune.py 

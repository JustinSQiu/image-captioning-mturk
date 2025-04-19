print('Loading libraries...', flush=True)

import os
os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

import glob
import warnings

import pandas as pd
import torch
import whisper
from transformers import pipeline

from huggingface_hub import login

login(token="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")

print('Loading models...', flush=True)

warnings.filterwarnings('ignore', message='FP16 is not supported on CPU')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}', flush=True)

tamil_transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-tamil-large-v2", chunk_length_s=30, device=device)
# tamil_transcribe.model.config.forced_decoder_ids = tamil_transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")
forced_decoder_ids = tamil_transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")
if forced_decoder_ids:  # Only assign if the list is not empty
    tamil_transcribe.model.config.forced_decoder_ids = forced_decoder_ids
else:
    tamil_transcribe.model.config.forced_decoder_ids = None  # Or simply omit this configuration

print(f'Transcribing audio...', flush=True)

audio_file = '/nlp/data/jsq/audio/1nxreKKxOBq4.mp3'

transcription = tamil_transcribe(audio_file)['text']

print(transcription)
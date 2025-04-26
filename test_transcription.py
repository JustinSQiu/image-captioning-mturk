print('Loading libraries...', flush=True)

import os
os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

import glob
import warnings

import pandas as pd
import torch
import whisper

from huggingface_hub import login
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline



login(token="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")

print('Loading models...', flush=True)

warnings.filterwarnings('ignore', message='FP16 is not supported on CPU')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f'Using device: {device}', flush=True)

# large_model = whisper.load_model('large-v3')
# turbo_model = whisper.load_model('turbo')

tamil_transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-tamil-large-v2", chunk_length_s=30, device=device)
tamil_transcribe.model.config.forced_decoder_ids = tamil_transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")

# viet_transcribe = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-large", device=device)

# nepali_transcribe = pipeline(model="kiranpantha/whisper-large-v3-turbo-nepali", device=device, tokenizer="openai/whisper-large-v2",) # doesn't work

# bengali_transcribe = pipeline("automatic-speech-recognition", model="KhushiDS/whisper-large-v3-Bengali", device=device)

# telugu_transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-telugu-large-v2", chunk_length_s=30, device=device)
# telugu_transcribe.model.config.forced_decoder_ids = telugu_transcribe.tokenizer.get_decoder_prompt_ids(language="te", task="transcribe")

# amharic_transcribe = pipeline("automatic-speech-recognition", model="drmeeseeks/whisper-large-v2-amet", device=device) # doesn't work

# thai_transcribe = pipeline("automatic-speech-recognition", model="biodatlab/whisper-th-large-combined", device=device)

# kinyarwanda_transcribe = pipeline("automatic-speech-recognition", model="dmusingu/WHISPER-SMALL-SWAHILI-ASR-CV-14", device=device) # doesn't work

print(f'Transcribing audio...', flush=True)

audio_file = '/nlp/data/jsq/audio/1nxreKKxOBq4.mp3' # Tamil
# audio_file = '/nlp/data/jsq/audio/14XUZFw5gvwB.mp3' # Viet
# audio_file = '/nlp/data/jsq/audio/1aaE9dN6bCXs.mp3' # Nepali
# audio_file = '/nlp/data/jsq/audio/16S3XoWRmuqT.mp3' # Bengali
# audio_file = '/nlp/data/jsq/audio/1mF5BP74i19R.mp3' # Telugu
# audio_file = '/nlp/data/jsq/audio/1fx1T2oLhZEe.mp3' # Amharic
# audio_file = '/nlp/data/jsq/audio/1dObTLHcjref.mp3' # Thai
# audio_file = '/nlp/data/jsq/audio/11JbdrbaOPzP.mp3' # Kinyarwanda

transcription = tamil_transcribe(audio_file)['text']

# transcription = turbo_model.transcribe(audio_file, language='nepali')['text']


print(transcription)
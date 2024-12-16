import os
import warnings

import pandas as pd
import whisper

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

model = whisper.load_model('turbo')

print(model.transcribe(audio='audio/1e6rzQlaiC5E.mp3', task='transcribe', language='hi'))

# import torch
# from transformers import pipeline

# # path to the audio file to be transcribed
# audio = "/path/to/audio.format"
# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-telugu-base", chunk_length_s=30, device=device)
# transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="te", task="transcribe")

# print('Transcription: ', transcribe(audio)["text"])

import os
import torch
import torchlibrosa
import librosa
from args import args
import json
import numpy as np
import random
import math
import os
import shutil
from utilities import ritorna_durata_audio_dataset
import speechbrain as sb
from speechbrain.inference.classifiers import AudioClassifier
path="C:/Users/aldob/Desktop/Dataset Progetto/TUT-sound-events-2017-development/audio/street"

dictionary_audio,max,min,media=ritorna_durata_audio_dataset(path)
#print(dictionary_audio,max,min,media)

'''
encoder = AudioClassifier.from_hparams(source="speechbrain/cnn14-esc50", savedir='pretrained_models/cnn14-esc50')
out_probs, score, index, text_lab = encoder.classify_file('speechbrain/cnn14-esc50/example_dogbark.wav')
print(encoder)  
'''
from transformers import Wav2Vec2Model

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
print(model)
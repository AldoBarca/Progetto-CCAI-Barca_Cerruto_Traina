import os
import torch
import torchlibrosa
import librosa
from args import args
import json
import numpy as np
import random
import math
from utilities import ritorna_durata_audio_dataset
path="C:/Users/aldob/Desktop/Dataset Progetto/TUT-sound-events-2017-development/audio/street"

dictionary_audio,max,min,media=ritorna_durata_audio_dataset(path)
print(dictionary_audio,max,min,media)

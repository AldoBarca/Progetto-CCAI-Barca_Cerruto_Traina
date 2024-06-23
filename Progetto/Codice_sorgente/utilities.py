import os
import torch
import torchlibrosa
import librosa
from args import args
import json
import numpy as np
import random
import math

def ritorna_nome_audio_dataset(path):
    try:
       
        items = os.listdir(path)
       
        files = [item for item in items if os.path.isfile(os.path.join(path, item)) and item.endswith('.wav')]
        
        return files
    except Exception as e:
        print(f"Errore: {e}")
        return []


def ritorna_durata_audio_dataset(path):
    lista_audio=ritorna_nome_audio_dataset(path)
    dictionary_audio={}
    durata_max=0
    durata_min=0
    numero_audio=0
    durata_totale=0
    for audio in lista_audio:
        audio_path=os.path.join(path,audio)
        audio_loaded, sr=librosa.load(audio_path)
        durata_audio=librosa.get_duration(y=audio_loaded, sr=sr)
        dictionary_audio[audio]=durata_audio
        if(durata_min==0):
            durata_min=durata_audio
        elif(durata_min>durata_audio):
            durata_min=durata_audio
        if(durata_max<durata_audio):
            durata_max=durata_audio
        numero_audio=numero_audio+1
        durata_totale=durata_totale+durata_audio
    media=durata_totale/numero_audio
    return dictionary_audio,durata_max,durata_min,media


def calcola_durata_audio(set):
    durata_max=0
    durata_min=0
    durata_totale=0
    numero_audio=0
    for classe,file in set.items():
        for audio,events in file.items():
            audio_path=os.path.join(args.traindir,audio)
            audio_loaded, sr=librosa.load(audio_path)
            durata_audio=librosa.get_duration(y=audio_loaded, sr=sr)
            if(durata_min==0):
                durata_min=durata_audio
            elif(durata_min>durata_audio):
                durata_min=durata_audio
            if(durata_max<durata_audio):
                durata_max=durata_audio
            numero_audio=numero_audio+1
            durata_totale=durata_totale+durata_audio
    media=durata_totale/numero_audio
    return durata_max,durata_min,media
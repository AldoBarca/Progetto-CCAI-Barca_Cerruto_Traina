import argparse
import os
parser = argparse.ArgumentParser()


parser.add_argument("--device", type=str, default='cuda:0') #device dove fare il training, di default cuda
parser.add_argument("--workers", type=int, default=4) #numero dei workers


parser.add_argument("--bs", type=int, default=128) #batch size, di default 128

parser.add_argument("--lr", type=float, default=1e-2) #learning rate, di default 0.01

parser.add_argument("--epochs", type=int, default=50) #numero di epoche per il training. In teoria il numero andrebbe scelto ad hoc per evitare l'overfitting.

# audio
parser.add_argument("--sr", type=int, default=22050) #sampling rate for audio
parser.add_argument("--len", type=int, default=200) #segment duration for training in ms

# mel spec parameters
parser.add_argument("--nmels", type=int, default=128) #number of mels
parser.add_argument("--nfft", type=int, default=512) #size of FFT
parser.add_argument("--hoplen", type=int, default=128) #hop between STFT windows
parser.add_argument("--fmax", type=int, default=11025) #fmax
parser.add_argument("--fmin", type=int, default=50) #fmin


args = parser.parse_args()
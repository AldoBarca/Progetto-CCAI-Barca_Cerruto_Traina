import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
#import utilities
from Roba_gia_presa.modelli import Cnn14,init_layer
import config



class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527
        
        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict


def train():

    # Arugments & parameters
    sample_rate = 22050
    window_size = 512
    hop_size =128
    mel_bins = 64 #128
    fmin = 50
    fmax = 11025
    model_type = "Transfer_Cnn14"
    pretrained_checkpoint_path = "C:/Users/aldob/Documents/GitHub/Thesis/Secondi_classificati_2023/CNN14/Codice sorgente/audioset_tagging_cnn-master/Cnn14_16k_mAP=0.438.pth"
    freeze_base = True
    device = 'cuda' if (torch.cuda.is_available()) else 'cpu'

    classes_num = 6
    pretrain = True if pretrained_checkpoint_path else False
    
    # Model
    Model = eval(model_type)
    encoder = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
        classes_num, freeze_base)
    
    # Load pretrained model
    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        encoder.load_from_pretrain(pretrained_checkpoint_path)

    print('GPU number: {}'.format(torch.cuda.device_count()))
   
    return encoder
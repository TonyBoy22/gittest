'''
Trains a network using the torch.util.data.Dataset Librispeech 
'''

from torch.utils.data import DataLoader
from torchaudio import datasets
import torchaudio
import torch
from torch import nn
import ssl
import os

from Model import EPOCHS, LEARNING_RATE, FeedForwardNet, train_one_epoch, train

if __name__ == "__main__":
    
    ssl._create_default_https_context = ssl._create_unverified_context
    
    torchaudio.set_audio_backend("soundfile")
    
    _SAMPLE_DIR = "_assets"
    LS_DATASET_PATH = os.path.join(_SAMPLE_DIR, "librispeech")
    os.makedirs(LS_DATASET_PATH, exist_ok=True)
    # Mettre un if/else pour le download selon la présence du dataset dans le repo
    librispeech_data = datasets.LIBRISPEECH(LS_DATASET_PATH, download=False)
    train_data_loader = DataLoader(librispeech_data, batch_size=1)
    
    # build model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f'using {device} device')
    
    feed_forward_net = FeedForwardNet().to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(), lr=LEARNING_RATE)

    # train model
    train(feed_forward_net, train_data_loader, loss_fn, optimiser, device, EPOCHS)
    
    # Store the model once it is trained
    torch.save(feed_forward_net.state_dict(), "librispeechnet.pth") # state_dict =
    print("Models trained and stored at librispeechnet.pth")
    
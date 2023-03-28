import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

from dataset import sample_rate

class ContrastiveModel(nn.Module):
    def __init__(self):
        super(ContrastiveModel, self).__init__()

        self.spectrogram = MelSpectrogram(sample_rate)

        self.conv = nn.Conv1d(128, 512, 101, padding=50)
        self.norm = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        size = x.size()
        x = x.reshape(size[0] * size[1], *size[2:])

        x = self.spectrogram(x)

        x = self.conv(x)

        x = self.norm(x)

        x = self.relu(x)

        x = self.dropout(x)
        
        # reshape again
        # x = x.reshape(size[0], size[1], 512, -1)

        # todo: attention pooling
        x = x.mean(2)

        norm = x.norm(dim=1, keepdim=True)
        x = x / norm

        return x
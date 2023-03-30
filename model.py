import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

from dataset import sample_rate

class ContrastiveModel(nn.Module):
    def __init__(self, num_classes):
        super(ContrastiveModel, self).__init__()

        self.spectrogram = MelSpectrogram(sample_rate)

        self.conv = nn.Conv1d(128, 512, 101, padding=50)
        self.norm = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        print(num_classes)
        self.project = nn.Linear(512, num_classes)

    def forward(self, x):

        size = x.size()
        x = x.reshape(size[0] * size[1], *size[2:])

        x = self.spectrogram(x)

        x = self.conv(x)

        x = self.norm(x)

        x = self.relu(x)

        x = self.dropout(x)
        
        # reshape again

        # todo: attention pooling
        embs = x.mean(2)


        norm = embs.norm(dim=1, keepdim=True)
        embs = embs / norm

        x = embs.reshape(size[0], size[1], 512, -1)

        preds = self.project(embs)

        return embs, preds
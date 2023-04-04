import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

from dataset import sample_rate

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv1d(in_dim, out_dim, kernel, padding=kernel//2)
        self.norm = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class RepConvBlock(nn.Module):
    def __init__(self, dim, reps, kernel=3):
        super(RepConvBlock, self).__init__()

        self.reps = nn.ModuleList([ConvBlock(dim, dim, kernel) for _ in range(reps)])

        self.repconv = nn.Conv1d(dim, dim, kernel, padding=kernel//2)
        self.repnorm = nn.BatchNorm1d(dim)

        self.skipconv = nn.Conv1d(dim, dim, kernel, padding=kernel//2)
        self.skipnorm = nn.BatchNorm1d(dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):

        repped = x
        for layer in self.reps:
            repped = layer(repped)

        repped = self.repconv(repped)
        repped = self.repnorm(repped)

        x = self.skipconv(x)
        x = self.skipnorm(x)

        x = x + repped

        x = self.relu(x)
        x = self.dropout(x)

        return x


class ContrastiveModel(nn.Module):
    def __init__(self, num_classes):
        super(ContrastiveModel, self).__init__()

        self.spectrogram = MelSpectrogram(sample_rate)

        self.in_proj = ConvBlock(128, 512, 5)

        self.reps = nn.ModuleList(RepConvBlock(512, 2) for _ in range(3))

        self.out_proj = nn.Linear(512, num_classes)

    def forward(self, x):

        size = x.size()
        x = x.reshape(size[0] * size[1], *size[2:])

        x = self.spectrogram(x)

        x = self.in_proj(x)

        for layer in self.reps:
            x = layer(x)

        # todo: attention pooling rather than mean
        embs = x.mean(2)

        norm = embs.norm(dim=1, keepdim=True)
        embs = embs / norm

        # reshape again
        x = embs.reshape(size[0], size[1], 512, -1)

        preds = self.out_proj(embs)

        return embs, preds
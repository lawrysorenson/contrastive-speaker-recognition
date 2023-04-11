import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram
import torch.nn.functional as F
import math

from dataset import sample_rate

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3):
        super(ConvBlock, self).__init__()

        self.depth_conv = nn.Conv1d(in_dim, out_dim, kernel, padding=kernel//2, groups=in_dim)
        self.time_conv = nn.Conv1d(out_dim, out_dim, 1)
        self.norm = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.time_conv(x)
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


class AttentionPooling(nn.Module):
    def __init__(self, dim, h=8):
        super(AttentionPooling, self).__init__()

        self.h = h
        self.d_k = dim//h
        
        self.query = nn.parameter.Parameter(torch.zeros(1, h, 1, self.d_k))
        self.ks = nn.Linear(dim, self.h * self.d_k)
        self.vs = nn.Linear(dim, self.h * self.d_k)

        self.w_o = nn.Linear(self.h * self.d_k, dim)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x):

        x = x.transpose(1, 2)

        batches = x.size(0)

        query = self.query
        key = self.ks(x).view(batches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.vs(x).view(batches, -1, self.h, self.d_k).transpose(1, 2)
    
        # scores = QK^T/scale
        # (baches, h, 1, d_k) matmul (batches, h, d_k, length) -> (batches, 8, 1, length)
        scores = query.matmul(key.transpose(2, 3)) / math.sqrt(self.d_k)

        # (batches, h, 1, length) matmul (batches, h, length, d_k) -> (batches, h, 1, d_k)
        output = F.softmax(scores, dim=3).matmul(value)

        # concatenate output from all heads
        # (batches, h, 1, d_k) -> (batches, h * d_k)
        output = output.reshape(batches, self.h * self.d_k)

        # (batches, h * d_k) -> (batches, dim)
        output = self.w_o(output)

        return output


class ContrastiveModel(nn.Module):
    def __init__(self, num_classes):
        super(ContrastiveModel, self).__init__()

        self.spectrogram = MelSpectrogram(sample_rate)

        self.in_proj = ConvBlock(128, 512, 9)

        self.reps = nn.ModuleList(RepConvBlock(512, 2, 7) for _ in range(3))

        self.pool = AttentionPooling(512)

        self.out_proj = nn.Linear(512, 1024)
        self.relu = nn.ReLU()
        self.out_label = nn.Linear(1024, 2)


    def forward(self, x):

        size = x.size()
        x = x.reshape(size[0] * size[1], *size[2:])

        x = self.spectrogram(x)

        x = self.in_proj(x)

        for layer in self.reps:
            x = layer(x)

        embs = self.pool(x)

        norm = embs.norm(dim=1, keepdim=True)
        embs = embs / norm

        # sims = torch.inner(embs, embs).unsqueeze(dim=2)

        # preds = self.out_proj(sims)

        # print(preds.size())

        out = self.out_proj(embs)
        
        out = out.unsqueeze(1) + out.unsqueeze(0)

        out = self.relu(out)
        out = self.out_label(out)

        return embs, out



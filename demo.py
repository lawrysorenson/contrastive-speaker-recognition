from dataset import ZipDataset, pad_to_longest
from model import ContrastiveModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torchaudio
from torchaudio.transforms import Resample
import torch.nn.functional as F
from dataset import wav_transform
import os

device = torch.device("cuda:0" if 'CUDA_VISIBLE_DEVICES' in os.environ else "cpu")

print('Using device', device)

model = ContrastiveModel(2)
model.load_state_dict(torch.load('best_model', map_location=device))

files = ['../voxceleb_trainer/data/wav/id10092/LbVIZMrQGmQ/00001.wav',
    '../voxceleb_trainer/data/wav/id10092/ifoQZ4hauRU/00001.wav',
    '../voxceleb_trainer/data/wav/id10098/a0yFzrtncuk/00005.wav',
    '../voxceleb_trainer/data/wav/id10098/jJLQc0W1fds/00006.wav'
]

tensors = []

for f in files:
    audio, sample_rate = torchaudio.load(f, format='wav')
    audio = wav_transform(audio, sample_rate)
    tensors.append(audio)


esims = [[0]*len(files) for _ in range(len(files))]
psims = [[0]*len(files) for _ in range(len(files))]
with torch.no_grad():
    model.eval()

    embs = []
    for t in tensors:
        t = t.unsqueeze(0)
        # print(t.size())
        # break
        emb, _ = model(t.to(device))
        embs.append(emb)

    for i in range(len(files)):
        for j in range(len(files)):
            ei = embs[i]
            ej = embs[j]
            sim = torch.inner(ei, ej).squeeze()
            esims[i][j] = sim.item()

    
    projs = []
    for t in tensors:
        t = t.unsqueeze(0)
        emb, _ = model(t.to(device))
        proj = model.out_proj(emb)
        projs.append(proj)

    for i in range(len(files)):
        for j in range(len(files)):
            ei = projs[i]
            ej = projs[j]

            comp = ei + ej
            comp = model.relu(comp)
            comp = model.out_label(comp)
            comp = F.softmax(comp).squeeze()
            psims[i][j] = comp[1].item()


for row in esims:
    for i in row:
        print(f'{i:10.6f}', end='')
    print()
print()

for row in psims:
    for i in row:
        print(f'{i:10.6f}', end='')
    print()


# model = model.to(device)

# with torch.no_grad():
#     model.eval()
#     n = len(test_dataloader)
#     progress = tqdm(total=n*(n-1)//2, desc=f'Test Fscore: -')
#     conf = [[0]*2 for _ in range(2)]
#     past = []
#     for x, y in test_dataloader:

#         x = x.to(device)
#         y = y.to(device)

#         embs, _ = model(x)
        
#         # for projection
#         proj = model.out_proj(embs)
#         #proj = embs

#         past.append((proj, y))

#         count = 0

#         for pp, py in past:

#             #comp = torch.inner(pp, proj).unsqueeze(2)

#             # for projection
#             comp = pp.unsqueeze(0) + proj.unsqueeze(1)
#             comp = model.relu(comp)

#             comp = model.out_label(comp)
#             comp = comp.argmax(2)    
#             count += comp.size(0)
#             comp = comp.reshape(-1)


#             labels = (py.unsqueeze(0) == y.unsqueeze(1)).long()
#             labels = labels.reshape(-1)


#             for a, b in zip(labels.tolist(), comp.tolist()):
#                 conf[a][b] += 1
#             precision = conf[1][1] / max(conf[0][1] + conf[1][1], 1)
#             recall = conf[1][1] / max(conf[1][0] + conf[1][1], 1)
#             fscore = 2 * precision * recall / max(precision + recall, 1)

#             progress.update(1)
#             progress.set_description(f'Test Fscore: {100*fscore:.4f}')
        

#     progress.close()

#     for row in conf:
#         print(row)

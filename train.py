from dataset import ZipDataset, pad_to_longest
from model import ContrastiveModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

device = torch.device("cuda:0" if 'CUDA_VISIBLE_DEVICES' in os.environ else "cpu")

print('Using device', device)

# dataset = ZipDataset(['vox1_dev_wav.zip', 'vox2_dev_wav.zip'])
dataset = ZipDataset(['vox1_dev_wav.zip'], test=False)
train_dataset, val_dataset = random_split(dataset, [0.85, 0.15])

train_dataloader = DataLoader(train_dataset, batch_size=10, collate_fn=pad_to_longest, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=10, collate_fn=pad_to_longest)
# dataset = ZipDataset(['vox2_dev_wav.zip'])

# test_dataset = ZipDataset(['vox1_test_wav.zip', 'vox2_test_wav.zip'], test=True)
# test_dataloader = DataLoader(test_dataset)


model = ContrastiveModel(dataset.num_classes)

def contrastive_loss(y):

    batch = y.size(0)

    sims = torch.inner(y, y)

    mask = torch.arange(batch)

    # mask diagonal
    sims[mask,mask] = -1000

    sims = sims / 0.2 # temperature

    ext = torch.arange(0, batch, 2)

    same = sims[ext, ext+1]
    same = torch.stack((same, same), dim=1)
    same = same.reshape(-1)

    losses = sims.logsumexp(1) - same

    return losses.mean(), sims.argmax(dim=1).floor_divide(2)

model = model.to(device)
model.train()

stabalize = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5)

for epoch in range(1, 100000):
    optimizer.zero_grad()

    # limit = (epoch + 19) // 20
    # i = 0
    # progress = tqdm(total=limit, desc=f'Train Epoch: {epoch} Loss: -')
    progress = tqdm(total=len(train_dataloader), desc=f'Train Epoch: {epoch} Loss: -')
    for x, y in train_dataloader:
        
        # double y for pairs
        cont_ys = torch.arange(y.size(0)).to(device)
        cont_ys = torch.stack([cont_ys, cont_ys], dim=1).reshape(-1)

        pair_labels = (cont_ys.unsqueeze(0) == cont_ys.unsqueeze(1)).long()
        pair_labels = pair_labels.reshape(-1)

        # y = torch.stack([y, y], dim=1).reshape(-1)
        # y = y.to(device)

        x = x.to(device)

        embs, pairs = model(x)
        pairs = pairs.reshape(-1, 2)

        loss, cont_preds = contrastive_loss(embs)
        loss = 0.5*loss + 0.5*stabalize(pairs, pair_labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        cont_accuracy = (cont_preds == cont_ys).float().mean().item()
        pair_accuracy = (pairs.argmax(1) == pair_labels).float().mean().item()

        progress.update(1)
        progress.set_description(f'Train E: {epoch} L: {loss.item():.4f} C-Acc: {cont_accuracy:.4f}  P-Acc: {pair_accuracy:.4f}')

        break

        # i += 1
        # if i == limit:
        #     break

        # break
    
    progress.close()


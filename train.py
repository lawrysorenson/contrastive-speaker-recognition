from dataset import ZipDataset, pad_to_longest
from model import ContrastiveModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# dataset = ZipDataset(['vox1_dev_wav.zip', 'vox2_dev_wav.zip'])
dataset = ZipDataset(['vox1_dev_wav.zip'], test=False)
train_dataloader = DataLoader(dataset, batch_size=10, collate_fn=pad_to_longest)
# dataset = ZipDataset(['vox2_dev_wav.zip'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ContrastiveModel(dataset.num_classes)

def contrastive_loss(y):
    # TODO: temperature

    batch = y.size(0)

    sims = torch.inner(y, y)

    mask = torch.arange(batch)

    # mask diagonal
    sims[mask,mask] = -1000

    ext = torch.arange(0, batch, 2)

    same = sims[ext, ext+1]
    same = torch.stack((same, same), dim=1)
    same = same.reshape(-1)

    losses = sims.logsumexp(1) - same

    return losses.mean()

model = model.to(device)
model.train()

stabalize = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5)

for epoch in range(1, 1000):
    optimizer.zero_grad()

    progress = tqdm(total=len(train_dataloader), desc=f'Train Epoch: {epoch} Loss: -')
    for x, y in train_dataloader:

        # double y for pairs
        y = torch.stack([y, y], dim=1).reshape(-1)

        x = x.to(device)
        y = y.to(device)

        embs, pred = model(x)

        # loss = contrastive_loss(pred)
        loss = stabalize(pred, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        accuracy = (pred.argmax(dim=1) == y).float().mean().item()

        progress.update(1)
        progress.set_description(f'Train Epoch: {epoch} Loss: {loss.item():.4f} Accuracy: {accuracy:.4f}')

        break
    
    progress.close()


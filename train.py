from dataset import ZipDataset, pad_to_longest
from model import ContrastiveModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# dataset = ZipDataset(['vox1_dev_wav.zip', 'vox2_dev_wav.zip'])
dataset = ZipDataset(['vox1_dev_wav.zip'], test=False)
train_dataloader = DataLoader(dataset, batch_size=10, collate_fn=pad_to_longest)
# dataset = ZipDataset(['vox2_dev_wav.zip'])

model = ContrastiveModel()

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

# model = model.cuda()
model.train()

objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5)

for epoch in range(1, 2):
    optimizer.zero_grad()

    for x in train_dataloader:
        # x = x.cuda()

        y = model(x)

        loss = contrastive_loss(y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(loss.item())

        # break


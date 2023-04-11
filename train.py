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

dataset = ZipDataset(['vox1_dev_wav.zip', 'vox2_dev_wav.zip'])
# dataset = ZipDataset(['vox1_dev_wav.zip'], test=False)
train_dataset, val_dataset = random_split(dataset, [0.85, 0.15])

batch_size = 10
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_to_longest, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4*batch_size, collate_fn=pad_to_longest)
# dataset = ZipDataset(['vox2_dev_wav.zip'])

# test_dataset = ZipDataset(['vox1_test_wav.zip', 'vox2_test_wav.zip'], test=True)
test_dataset = ZipDataset(['vox1_test_wav.zip'], test=True)
test_dataloader = DataLoader(test_dataset, batch_size=8*batch_size, collate_fn=pad_to_longest)


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

class_weights = torch.tensor([1, batch_size/5])
class_weights = class_weights.to(device)
stabalize = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=3e-5)

early_stop = 0
best_model = None
best_score = -1

for epoch in range(1, 100000):

    model.train()
    optimizer.zero_grad()
    progress = tqdm(total=len(train_dataloader), desc=f'Train E: {epoch} L: - C-Acc: - Fscore: -')
    i = 0
    for x, y in train_dataloader:
        i += 1
        if i == 11:
            break
        
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
        conf = [[0]*2 for _ in range(2)]
        for a, b in zip(pair_labels.tolist(), pairs.argmax(1).tolist()):
            conf[a][b] += 1
        precision = conf[1][1] / max(conf[0][1] + conf[1][1], 1)
        recall = conf[1][1] / max(conf[1][0] + conf[1][1], 1)
        fscore = 2 * precision * recall / max(precision + recall, 1)
        
        progress.update(1)
        progress.set_description(f'Train E: {epoch} L: {loss.item():.4f} C-Acc: {100*cont_accuracy:.2f} Fscore: {100*fscore:.2f}')

        # break
    
    progress.close()

    with torch.no_grad():
        model.eval()
        progress = tqdm(total=len(val_dataloader), desc=f'Val   E: {epoch} L: - C-Acc: - Fscore: -')
        cacc_num = 0
        cacc_tot = 0
        conf = [[0]*2 for _ in range(2)]
        i = 0
        for x, y in val_dataloader:
            
            i += 1
            if i == 11:
                break

            cont_ys = torch.arange(y.size(0)).to(device)
            cont_ys = torch.stack([cont_ys, cont_ys], dim=1).reshape(-1)

            pair_labels = (cont_ys.unsqueeze(0) == cont_ys.unsqueeze(1)).long()
            pair_labels = pair_labels.reshape(-1)

            x = x.to(device)

            embs, pairs = model(x)
            pairs = pairs.reshape(-1, 2)

            _, cont_preds = contrastive_loss(embs)

            cacc_num += (cont_preds == cont_ys).long().sum().item()
            cacc_tot += cont_ys.size(0)

            for a, b in zip(pair_labels.tolist(), pairs.argmax(1).tolist()):
                conf[a][b] += 1
            precision = conf[1][1] / max(conf[0][1] + conf[1][1], 1)
            recall = conf[1][1] / max(conf[1][0] + conf[1][1], 1)
            fscore = 2 * precision * recall / max(precision + recall, 1)

            progress.update(1)
            progress.set_description(f'Val   E: {epoch} L: {loss.item():.4f} C-Acc: {100*cacc_num/cacc_tot:.2f} Fscore: {100*fscore:.2f}')

            # break
        
        progress.close()


        print('Confusion matrix')
        for row in conf:
            print(row)

        if fscore > best_score + 0.0001:
            print('Saving model with fscore', fscore)
            best_score = fscore
            early_stop = 0
            best_model = model.state_dict()
            torch.save(best_model, 'best_model')
            break # remove
        else:
            early_stop += 1
            if early_stop >= 20:
                break
            

# Load the best model
model.load_state_dict(best_model)

with torch.no_grad():
    model.eval()
    n = len(test_dataloader)
    progress = tqdm(total=n*(n-1)//2, desc=f'Test Fscore: -')
    conf = [[0]*2 for _ in range(2)]
    past = []
    for x, y in test_dataloader:
        print(y)

        x = x.to(device)
        y = y.to(device)

        embs, _ = model(x)

        proj = model.out_proj(embs)

        past.append((proj, y))

        count = 0

        for pp, py in past:

            comp = pp.unsqueeze(0) + proj.unsqueeze(1)
            comp = model.relu(comp)
            comp = model.out_label(comp)
            comp = comp.argmax(2)    
            count += comp.size(0)
            comp = comp.reshape(-1)


            labels = (py.unsqueeze(0) == y.unsqueeze(1)).long()
            labels = labels.reshape(-1)

            print(comp)
            print(labels)

            #for a, b in zip(pair_labels.tolist(), pairs.argmax(1).tolist()):
            #    conf[a][b] += 1
            #precision = conf[1][1] / max(conf[0][1] + conf[1][1], 1)
            #recall = conf[1][1] / max(conf[1][0] + conf[1][1], 1)
            #fscore = 2 * precision * recall / max(precision + recall, 1)

            break


            
            progress.update(1)
            #progress.set_description(f'Test Fscore: {100*fscore:.2f}')
        
        exit(0)
#         pair_labels = (cont_ys.unsqueeze(0) == cont_ys.unsqueeze(1)).long()
#         pair_labels = pair_labels.reshape(-1)

#         x = x.to(device)


    
    progress.close()

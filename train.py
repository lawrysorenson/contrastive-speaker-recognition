from dataset import ZipDataset

dataset = ZipDataset(['vox1_dev_wav.zip', 'vox2_dev_aac.zip'])

for out in dataset:
    print(out)
    break


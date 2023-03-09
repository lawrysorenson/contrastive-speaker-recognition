from zipfile import ZipFile
import torch
import torchaudio
from torch.utils.data import Dataset
import random
from tqdm import tqdm

data_path = '../voxceleb_trainer/data/'

class ZipDataset(Dataset):
    def __init__(self, files, test=False):

        def find_id(path):
            for p in path.split('/'):
                if p.startswith('id'):
                    return p
            raise Exception('No id found in path')

        self.zfs = []
        self.ids = {}
        self.idx2ids = []
        self.paths = []
        self.test = test

        for i, path in enumerate(files):
            zf = ZipFile(data_path + path, 'r')
            self.zfs.append(zf)
            print('Loading dataset',path)

            sub = zf.namelist()
            for wav in tqdm(sub, ):
                if not (wav.endswith('wav') or wav.endswith('m4a')):
                    continue

                fid = find_id(wav)
                if fid not in self.ids:
                    fin = len(self.paths)
                    self.paths.append([])
                    self.ids[fid] = (fin, zf)
                    self.idx2ids.append(fid)
                else:
                    fin, zfc = self.ids[fid]
                    assert zf is zfc
                self.paths[fin].append(wav)
                #break

        #print(len(self.paths))
        #print(sum(len(i) for i in self.paths))
        #print(max(len(i) for i in self.paths))

        if self.test:
            # flatten
            new_paths = []
            for fid, (fin, zfc) in self.ids.items():
                for path in self.paths[fin]:
                    new_paths.append((path, fid))
                
                self.ids[fid] = zfc

            self.paths = new_paths


    def __del__(self):
        if self.zfs:
            for zf in self.zfs:
                #print('Closing',zf)
                zf.close()


    def __len__(self):
        return len(self.paths)

    
    def _read(self, zf, path):
        # select random snippet or pad to length
        audio = zf.open(path)
        audio, sample_rate = torchaudio.load(audio)
        print(audio.size(), sample_rate)
        return 0
    

    def getitem_train(self, idx):
        fid = self.idx2ids[idx]
        _, zf = self.ids[fid]
        sel = random.choices(self.paths[idx], k=2)

        out = [self._read(zf, path) for path in sel]

        #print(out)
        return 0
        pass


    def getitem_test(self, idx):
        return 1
        #pass

    def __getitem__(self, idx):
        return self.getitem_test(idx) if self.test else self.getitem_train(idx)

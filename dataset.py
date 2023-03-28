from zipfile import ZipFile
import torch
import torchaudio
from torchaudio.transforms import Resample
from torch.utils.data import Dataset
import random
from tqdm import tqdm
from subprocess import Popen, PIPE, STDOUT
import io

data_path = '../voxceleb_trainer/data/'
sample_rate = 8000

def pad_to_longest(batch):
    pairs = type(batch[0]) == tuple
    if pairs:
        first, last = zip(*batch)
        batch = list(first) + list(last)
    
    pad_len = max(a.size(1) for a in batch)

    pad_batch = [torch.cat((a, torch.zeros(1, pad_len - a.size(1))), dim=1) for a in batch]
    
    if pairs:
        n = len(pad_batch) // 2
        first = pad_batch[:n]
        last = pad_batch[n:]
        pad_batch = [torch.cat((a, b), dim=0) for a, b in zip(first, last)]

    pad_batch = torch.stack(pad_batch, dim=0)

    return pad_batch   

def wav_transform(audio, orig_sample):
    resample = Resample(orig_freq=orig_sample, new_freq=sample_rate)
    resampled = resample(audio)

    audio_end = resampled.size(1)

    # sample length
    sample_length = 10 * sample_rate # up to ten seconds
    
    start = random.randint(0, max(audio_end - sample_length, 0))

    sampled = resampled[:,start:start+sample_length]

    return sampled

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
            for wav in tqdm(sub):
                if not (wav.endswith('wav') or wav.endswith('m4a')):
                    continue
                # print(wav, flush=True)

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
                
                # break # remove this line

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
        audio, sample_rate = torchaudio.load(audio, format='wav')
        audio = wav_transform(audio, sample_rate)
        return audio
    

    def getitem_train(self, idx):
        fid = self.idx2ids[idx]
        _, zf = self.ids[fid]
        sel = random.choices(self.paths[idx], k=2)

        out = [self._read(zf, path) for path in sel]

        return tuple(out)

    def getitem_test(self, idx):
        path, fid = self.paths[idx]
        zf = self.ids[fid]
        out = self._read(zf, path)
        return out

    def __getitem__(self, idx):
        return self.getitem_test(idx) if self.test else self.getitem_train(idx)

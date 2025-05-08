import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import librosa
import numpy as np

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

class TESSDataset(Dataset):
    def __init__(self, root_dir, max_len=128):
        self.root_dir = root_dir
        self.max_len = max_len
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.dataset = datasets.DatasetFolder(root_dir, loader=self._load_audio, extensions=('.wav',))

    def _load_audio(self, filepath):
        audio, sr = librosa.load(filepath, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        if mfcc.shape[1] < self.max_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, self.max_len - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :self.max_len]
        return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

dataset = TESSDataset('../data/tess')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

if __name__ == "__main__":
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
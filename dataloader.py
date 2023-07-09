import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.block_size = block_size

        self.data = text
        # with open(text, 'r') as f:
        #   self.data = f.read()

        self.len = len(self.data) - self.block_size - 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx:idx+self.block_size], self.data[idx+1:idx+self.block_size+1]


def get_dataloader(text, batch_size=10, block_size=8, split=0.9):

    # with open(text_file, 'r') as f:
    #     text = f.read()

    n = int(len(text) * split)

    train = TextDataset(text[:n], block_size=block_size)
    test = TextDataset(text[n:], block_size=block_size)

    train_loader = DataLoader(train, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    return train_loader, test_loader

import random
import h5py
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, min_idx, excluded_max, reader):
        self.reader = reader
        self.min = min_idx
        self.max = excluded_max

    def __len__(self):
        return self.max - self.min

    def __getitem__(self, idx):
        if self.min + idx >= self.max:
            raise Exception("Out of Bounds")
        datapoint = self.reader[self.min + idx]
        x = datapoint["X"][:7]
        y = datapoint["Y"][:][:]
        return x, y


class H5Reader:
    def __init__(self, hdf5_path):
        self.file = h5py.File(hdf5_path, "r")
        self.key_list = list(self.file.keys())
        random.shuffle(self.key_list)

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        if idx >= len(self.key_list):
            raise Exception("Out of Bounds")
        return self.file[self.key_list[idx]]


def train_test_split(anteil_test, hdf5_path):
    reader = H5Reader(hdf5_path)
    split_index = int(len(reader) * (1 - anteil_test))
    return CustomDataset(0, split_index, reader), CustomDataset(split_index, len(reader), reader)


train_dataset, test_dataset = train_test_split(1./3, "data.h5")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

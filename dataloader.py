# dataloader.py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch

class RSSIDataset(Dataset):
    def __init__(self, data_array):
        self.X = data_array[:, :-3].astype(np.float32)         # RSSI features
        self.y = data_array[:, -3:-1].astype(np.float32)       # (x, y)
        self.X = (self.X + 100)/100
        original_floor = data_array[:, -1].astype(np.int64)

        # 显式映射：3 -> 0, 5 -> 1
        self.floor_map = {3: 0, 5: 1}
        self.floor = np.vectorize(self.floor_map.get)(original_floor)

        self.num_floors = 2  # 显式设置为2
  

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.floor[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

def load_dataset(processed_file, val_split=0.2, batch_size=64, shuffle=True, seed=42):
    if processed_file.endswith(".csv"):
        data = pd.read_csv(processed_file, header=None).values
    elif processed_file.endswith(".npy"):
        data = np.load(processed_file)
    else:
        raise ValueError("Unsupported file type. Use .csv or .npy")

    dataset = RSSIDataset(data)

    # reproducible split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, worker_init_fn=lambda id: np.random.seed(seed + id)
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=0, worker_init_fn=lambda id: np.random.seed(seed + id)
    )

    return train_loader, val_loader, dataset.num_floors

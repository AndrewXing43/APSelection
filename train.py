# train_kfold.py
import torch
import numpy as np
import random
import os
from sklearn.model_selection import KFold
from CNN import RSSICNN
from utils import get_loss_fn, compute_rmse, save_model
from dataloader import RSSIDataset
import pandas as pd

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_kfold(
    data_path,
    k=5,
    num_epochs=1000,
    batch_size=64,
    learning_rate=1e-3,
    loss_type="smoothl1",
    floor_embed_dim=8,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir="checkpoints_kfold"
):
    set_seed(42)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Device] Using {device}")

    # Load dataset
    if data_path.endswith(".csv"):
        data = pd.read_csv(data_path, skiprows=1, header=None).values

    else:
        raise ValueError("Only .csv supported")

    full_dataset = RSSIDataset(data)
    X = data[:, :-3].astype(np.float32)
    y = data[:, -3:-1].astype(np.float32)
    floor_ids = data[:, -1].astype(np.int64)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n=== Fold {fold}/{k} ===")

        train_data = data[train_idx]
        val_data = data[val_idx]

        train_set = RSSIDataset(train_data)
        val_set = RSSIDataset(val_data)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                   worker_init_fn=lambda id: np.random.seed(42 + id))
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

        num_floors = train_set.num_floors  # Assume same mapping as val set

        model = RSSICNN(num_floors=num_floors, floor_embed_dim=floor_embed_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = get_loss_fn(loss_type)

        best_val_rmse = float("inf")
        best_model_path = os.path.join(save_dir, f"fold{fold}_best_model.pth")

        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0.0
            for rssi, floor, coords in train_loader:
                rssi, floor, coords = rssi.to(device), floor.to(device), coords.to(device)
                optimizer.zero_grad()
                preds = model(rssi, floor)
                loss = criterion(preds, coords)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * rssi.size(0)

            avg_train_loss = total_loss / len(train_loader.dataset)

            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for rssi, floor, coords in val_loader:
                    rssi, floor, coords = rssi.to(device), floor.to(device), coords.to(device)
                    preds = model(rssi, floor)
                    val_preds.append(preds.cpu())
                    val_targets.append(coords.cpu())

            val_preds = torch.cat(val_preds, dim=0)
            val_targets = torch.cat(val_targets, dim=0)
            val_rmse = compute_rmse(val_preds, val_targets)
            ale = torch.norm(val_preds - val_targets, dim=1).mean().item()

            if epoch % 50 == 0 or epoch == 1:
                print(f"Epoch {epoch:04d} | Train Loss: {avg_train_loss:.4f} | Val RMSE: {val_rmse:.4f} | ALE: {ale:.4f}")

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                save_model(model, best_model_path)
                print(f"  > New best model saved (Fold {fold}) with RMSE: {best_val_rmse:.4f}")

        fold_results.append((best_val_rmse, ale))

    print("\n==== Summary ====")
    for i, (rmse, ale) in enumerate(fold_results, 1):
        print(f"Fold {i}: RMSE = {rmse:.4f}, ALE = {ale:.4f}")
    avg_rmse = np.mean([r for r, _ in fold_results])
    avg_ale = np.mean([a for _, a in fold_results])
    print(f"\nAverage RMSE = {avg_rmse:.4f} | Average ALE = {avg_ale:.4f}")

if __name__ == "__main__":
    train_kfold(
        data_path="train_dataset.csv",
        k=5,
        num_epochs=600,
        batch_size=64,
        floor_embed_dim=8,
        learning_rate=1e-3,
        save_dir="checkpoints_kfold1"
    )

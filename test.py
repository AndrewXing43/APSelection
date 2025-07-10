import torch
import pandas as pd
import numpy as np
from CNN import RSSICNN
from utils import load_model, compute_rmse
from dataloader import RSSIDataset

def evaluate(model_path, test_csv, output_path="predictions.csv", device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"[Device] Using {device}")

    # 加载并标准化 test 数据
    test_data = pd.read_csv(test_csv, skiprows = 1,header=None).values
    test_dataset = RSSIDataset(test_data)

    # 构造 dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 加载模型（楼层数自动从 dataset 获取）
    model = RSSICNN(num_floors=test_dataset.num_floors, floor_embed_dim=8)
    model = load_model(model, model_path, device)
    model.eval()

    preds_all = []
    gts_all = []

    with torch.no_grad():
        for rssi, floor, coords in test_loader:
            rssi, floor = rssi.to(device), floor.to(device)
            preds = model(rssi, floor)
            preds_all.append(preds.cpu().numpy())
            gts_all.append(coords.numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    gts_all = np.concatenate(gts_all, axis=0)

    # 输出 RMSE
    rmse = np.sqrt(np.mean((preds_all - gts_all) ** 2))
    print(f"[Eval] RMSE = {rmse:.4f}")

    # 输出 ALE
    ale = np.mean(np.linalg.norm(preds_all - gts_all, axis=1))
    print(f"[Eval] ALE  = {ale:.4f}")

if __name__ == "__main__":
    evaluate(
        model_path="checkpoints_kfold/fold5_best_model.pth",
        test_csv="test_dataset.csv",
        output_path="predictions.csv"
    )

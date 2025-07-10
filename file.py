import pandas as pd

# 加载 CSV 文件
df = pd.read_csv("processed_dataset.csv", header=None)

# 将最后一列转换为整数
df.iloc[:, -1] = df.iloc[:, -1].astype(int)

# 保存为新文件（或覆盖原文件）
df.to_csv("train_dataset.csv", header=False, index=False)

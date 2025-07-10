import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class RSSICNN(nn.Module):
    def __init__(self, num_floors=10, floor_embed_dim=8):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.se = SEBlock(128)

        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)

        self.embedding = nn.Embedding(num_floors, floor_embed_dim)

        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(128 * 2 + floor_embed_dim, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, rssi, floor_id):
        x = rssi.unsqueeze(1)  # (B, 1, 620)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.conv3(x))
        x = self.se(x)

        x_avg = self.pool_avg(x).squeeze(2)  # (B, 128)
        x_max = self.pool_max(x).squeeze(2)  # (B, 128)
        cnn_feat = torch.cat([x_avg, x_max], dim=1)  # (B, 256)

        floor_embed = self.embedding(floor_id)  # (B, D)
        x = torch.cat([cnn_feat, floor_embed], dim=1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # (B, 2)

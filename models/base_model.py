import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CustomGELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.gelu(x, approximate='tanh')

class BaseConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            CustomGELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            CustomGELU()
        )

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class BaseModel(nn.Module):
    def __init__(self, in_dim, model_dim, drop_rate, modality="V"):
        super().__init__()
        dim = model_dim
        self.modality = modality
        self.embedding = nn.Sequential(
            nn.Conv1d(in_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        self.stage1 = BaseConvBlock(dim)
        self.stage2 = BaseConvBlock(dim)
        self.stage3 = BaseConvBlock(dim)
        self.pool = nn.AvgPool1d(2, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Conv1d(dim, 1, 1),
            nn.Sigmoid()
        )
        self.mse = nn.MSELoss()

    def forward(self, feats):
        x = feats[self.modality]
        if len(x.shape) == 4:
            x = x.mean(dim=2)
        x = rearrange(x, 'b t d -> b d t').contiguous()
        x = self.embedding(x)

        x1 = self.stage1(x)
        x1 = self.pool(x1)

        x2 = self.stage2(x1)
        x2 = self.pool(x2)

        x3 = self.stage3(x2)
        x3 = self.gap(x3)

        score = self.fc(x3).squeeze(dim=2)
        return score, {'feats': [x, x1, x2, x3]}

    def call_loss(self, pred, label, **kwargs):
        return self.mse(pred.squeeze(), label.squeeze())

# 在文件末尾添加安全全局变量
torch.serialization.add_safe_globals([BaseConvBlock, BaseModel, CustomGELU]) 
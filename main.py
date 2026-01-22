import os
# 设置OpenMP环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from datasets import AQADataset
from models.base_model import BaseModel
from models.shmap import SHMAP


def train_epoch(model, optimizer, dataloader):
    model.train()
    losses = []
    preds = []
    labels = []
    start_time = time.time()
    
    for video_data, audio_data, flow_data, label, label_grade in dataloader:
        video_data = video_data.cuda()
        audio_data = audio_data.cuda()
        flow_data = flow_data.cuda()
        label = label.cuda().float()
        label_grade = label_grade.cuda().long()
        
        input_feats = {
            'V': video_data,
            'F': flow_data,
            'A': audio_data
        }
        
        optimizer.zero_grad()
        output, other_info = model(input_feats)
        loss = model.call_loss(output, label, label_grade=label_grade, **other_info)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        preds.extend(output.cpu().detach().numpy().tolist())
        labels.extend(label.cpu().detach().numpy().tolist())
    
    elapsed_time = time.time() - start_time
    avg_loss = np.mean(losses) * 100
    coef, _ = spearmanr(preds, labels)
    
    return elapsed_time, avg_loss, coef


def evaluate(model, dataloader):
    model.eval()
    losses = []
    preds = []
    labels = []
    start_time = time.time()
    
    with torch.no_grad():
        for video_data, audio_data, flow_data, label, label_grade in dataloader:
            video_data = video_data.cuda()
            audio_data = audio_data.cuda()
            flow_data = flow_data.cuda()
            label = label.cuda().float()
            label_grade = label_grade.cuda().long()
            
            input_feats = {
                'V': video_data,
                'F': flow_data,
                'A': audio_data
            }
            
            output, other_info = model(input_feats)
            loss = model.call_loss(output, label, label_grade=label_grade, **other_info)
            
            losses.append(loss.item())
            preds.append(output.squeeze().cpu().detach().item())
            labels.append(label.cpu().detach().item())
    
    elapsed_time = time.time() - start_time
    avg_loss = np.mean(losses) * 100
    coef, _ = spearmanr(preds, labels)
    
    return elapsed_time, avg_loss, coef * 100


if __name__ == '__main__':
    import sys
    
    gpu = sys.argv[2] if len(sys.argv) > 2 else '0'
    action = sys.argv[4] if len(sys.argv) > 4 else 'Ball'
    feat_dir = './data/features'

    if gpu != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    lr = 5e-4
    batch_size = 10
    epochs = 100
    
    rgb_feat = "VST"
    flow_feat = "I3D"
    audio_feat = "AST"
    
    print(f"\nLoading {action} dataset...")
    train_dataset = AQADataset(
        dataset_name=action,
        feat_dir=feat_dir,
        rgb_feat=rgb_feat,
        flow_feat=flow_feat,
        audio_feat=audio_feat,
        squeeze_rgb_feat="mean",
        is_train=True
    )
    test_dataset = AQADataset(
        dataset_name=action,
        feat_dir=feat_dir,
        rgb_feat=rgb_feat,
        flow_feat=flow_feat,
        audio_feat=audio_feat,
        squeeze_rgb_feat="mean",
        is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    print(f"\nBuilding SHMAP model...")
    model = SHMAP(
        model_dim=256,
        fc_drop=0,
        fc_r=2,
        feat_drop=0.5,
        K=10 if action in ["Ball", "Clubs", "Hoop", "Ribbon"] else 6,
        ms_heads=1,
        cm_heads=1,
        ckpt_dir=None,
        dataset_name=action
    ).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    max_coef = 0
    max_avg_coef = 0
    min_loss = 10
    min_avg_loss = 10
    train_losses = []
    test_losses = []
    train_coefs = []
    test_coefs = []
    
    for epoch in range(epochs):
        train_time, train_loss, train_coef = train_epoch(model, optimizer, train_loader)
        test_time, test_loss, test_coef = evaluate(model, test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_coefs.append(train_coef)
        test_coefs.append(test_coef)
        
        max_coef = max(max_coef, test_coef)
        max_avg_coef = max(max_avg_coef, np.mean(test_coefs[-10:]))
        min_loss = min(min_loss, test_loss)
        min_avg_loss = min(min_avg_loss, np.mean(test_losses[-10:]))
        
        print(f"Epoch[{epoch}/{epochs}]\t"
              f"Time: {train_time:.1f}/{test_time:.1f}\t"
              f"Loss {train_loss:.4f}/{test_loss:.4f}\t"
              f"Avg_loss {np.mean(test_losses[-10:]):.2f}/{min_avg_loss:.2f}\t"
              f"Coef {train_coef:.2f}/{test_coef:.2f}\t"
              f"Avg_coef {np.mean(test_coefs[-10:]):.2f}/{max_avg_coef:.2f}\t"
              f"BestCoef {max_coef:.2f}\tBestLoss {min_loss:.2f}")
    
    print(f"\nTraining completed.")
    print(f"Best avg coef: {max_avg_coef:.4f}\tBest coef: {max_coef:.4f}")

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from models.shmap import SHMAP


ACTION_CONFIGS = {
    'Ball': {'epochs': 400, 'lr': 5e-4},
    'Clubs': {'epochs': 500, 'lr': 5e-4},
    'Hoop': {'epochs': 300, 'lr': 5e-4},
    'Ribbon': {'epochs': 500, 'lr': 5e-4},
    'TES': {'epochs': 300, 'lr': 5e-4},
    'PCS': {'epochs': 300, 'lr': 5e-4},
}


class RandomDataset:
    def __init__(self, num_samples, seq_len_range=(70, 120)):
        self.num_samples = num_samples
        self.seq_len_range = seq_len_range
        self.data = []
        self.labels = []
        
        for i in range(num_samples):
            seq_len = np.random.randint(*seq_len_range)
            rgb = torch.randn(seq_len, 1024)
            flow = torch.randn(seq_len, 1024)
            audio = torch.randn(seq_len, 768)
            label = float(np.random.rand() * 0.65 + 0.3)
            
            self.data.append({'rgb': rgb, 'flow': flow, 'audio': audio})
            self.labels.append(label)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def collate_fn(batch):
    data_list, labels = zip(*batch)
    
    max_len = max([d['rgb'].shape[0] for d in data_list])
    
    rgb_batch = []
    flow_batch = []
    audio_batch = []
    
    for data in data_list:
        rgb = data['rgb']
        flow = data['flow']
        audio = data['audio']
        
        if rgb.shape[0] < max_len:
            pad_len = max_len - rgb.shape[0]
            rgb = torch.cat([rgb, torch.zeros(pad_len, 1024)], dim=0)
            flow = torch.cat([flow, torch.zeros(pad_len, 1024)], dim=0)
            audio = torch.cat([audio, torch.zeros(pad_len, 768)], dim=0)
        
        rgb_batch.append(rgb)
        flow_batch.append(flow)
        audio_batch.append(audio)
    
    return {
        'rgb': torch.stack(rgb_batch),
        'flow': torch.stack(flow_batch),
        'audio': torch.stack(audio_batch),
        'label': torch.tensor(labels, dtype=torch.float32)
    }


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    epoch_loss = 0.0
    predictions_list = []
    labels_list = []
    
    for batch in dataloader:
        input_feats = {
            'rgb': batch['rgb'].to(device),
            'flow': batch['flow'].to(device),
            'audio': batch['audio'].to(device)
        }
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        predictions, _ = model(input_feats)
        
        loss = nn.MSELoss()(predictions, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        predictions_list.extend(predictions.detach().cpu().numpy())
        labels_list.extend(labels.cpu().numpy())
    
    avg_loss = epoch_loss / len(dataloader)
    rho, _ = spearmanr(predictions_list, labels_list)
    
    return avg_loss, rho * 100


def evaluate(model, dataloader, device):
    model.eval()
    epoch_loss = 0.0
    predictions_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_feats = {
                'rgb': batch['rgb'].to(device),
                'flow': batch['flow'].to(device),
                'audio': batch['audio'].to(device)
            }
            labels = batch['label'].to(device)
            
            predictions, _ = model(input_feats)
            loss = nn.MSELoss()(predictions, labels)
            
            epoch_loss += loss.item()
            predictions_list.extend(predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    avg_loss = epoch_loss / len(dataloader)
    rho, _ = spearmanr(predictions_list, labels_list)
    
    return avg_loss, rho * 100


def main():
    parser = argparse.ArgumentParser(description='SHMAP Training')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--action', type=str, required=True, 
                        choices=['Ball', 'Clubs', 'Hoop', 'Ribbon', 'TES', 'PCS'],
                        help='Action type')
    args = parser.parse_args()
    
    if args.gpu != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = ACTION_CONFIGS[args.action]
    num_epochs = config['epochs']
    lr = config['lr']
    
    print(f"Training {args.action} for {num_epochs} epochs on {device}")
    print("=" * 80)
    
    train_dataset = RandomDataset(num_samples=280)
    test_dataset = RandomDataset(num_samples=60)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10, shuffle=False, collate_fn=collate_fn
    )
    
    model = SHMAP(
        model_dim=256,
        fc_drop=0,
        fc_r=2,
        feat_drop=0.5,
        K=10,
        ms_heads=1,
        cm_heads=1
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_avg_coef = -1.0
    best_coef = -1.0
    min_avg_loss = float('inf')
    min_loss = float('inf')
    saved_model_coef = 0.0
    
    train_loss_history = []
    train_coef_history = []
    test_loss_history = []
    test_coef_history = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        train_loss, train_coef = train_epoch(model, train_loader, optimizer, device)
        
        test_loss, test_coef = evaluate(model, test_loader, device)
        
        epoch_time = time.time() - epoch_start_time
        
        train_loss_history.append(train_loss)
        train_coef_history.append(train_coef)
        test_loss_history.append(test_loss)
        test_coef_history.append(test_coef)
        
        avg_train_loss = np.mean(train_loss_history[-10:]) if len(train_loss_history) >= 10 else np.mean(train_loss_history)
        avg_test_loss = np.mean(test_loss_history[-10:]) if len(test_loss_history) >= 10 else np.mean(test_loss_history)
        avg_train_coef = np.mean(train_coef_history[-10:]) if len(train_coef_history) >= 10 else np.mean(train_coef_history)
        avg_test_coef = np.mean(test_coef_history[-10:]) if len(test_coef_history) >= 10 else np.mean(test_coef_history)
        
        if test_coef > best_coef:
            best_coef = test_coef
        if avg_test_coef > best_avg_coef:
            best_avg_coef = avg_test_coef
            saved_model_coef = test_coef
            torch.save(model.state_dict(), f'{args.action}_best.pth')
        if test_loss < min_loss:
            min_loss = test_loss
        if avg_test_loss < min_avg_loss:
            min_avg_loss = avg_test_loss
        
        print(f"Epoch[{epoch+1}/{num_epochs}]\t"
              f"Time: {epoch_time:.1f}/{epoch_time:.1f}\t"
              f"Loss {train_loss:.4f}/{test_loss:.4f}\t\t"
              f"Avg_loss {avg_train_loss:.2f}/{avg_test_loss:.2f}\t\t"
              f"Coef {train_coef:.2f}/{test_coef:.2f}\t\t"
              f"Avg_coef {avg_train_coef:.2f}/{avg_test_coef:.2f}\t"
              f"BestCoef {best_coef:.2f}\t"
              f"BestLoss {min_loss:.2f}")
    
    print("=" * 80)
    print(f"best avg coef: {best_avg_coef/100:.4f}\t"
          f"best coef:{best_coef/100:.4f}\t\t"
          f"min avg loss: {min_avg_loss:.6f}\t"
          f"min loss:{min_loss:.6f}\t\t"
          f"coef of saved model: {saved_model_coef/100:.4f}")
    print(f"{best_avg_coef:.4f}\t{best_coef:.4f}\t{min_avg_loss:.4f}\t{min_loss:.4f}")


if __name__ == '__main__':
    main()

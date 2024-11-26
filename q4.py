import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        logits = ast.literal_eval(self.data.iloc[idx]["Logits"])
        targets = list(map(int, self.data.iloc[idx]["Targets"].split(','))) 
        input_lengths = int(self.data.iloc[idx]["Input_Lengths"])
        target_lengths = int(self.data.iloc[idx]["Target_Lengths"])
        
        logits = torch.tensor(logits, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.long)
        
        return logits, targets, input_lengths, target_lengths


def collate_fn(batch):
    logits_batch, targets_batch, input_lengths_batch, target_lengths_batch = zip(*batch)
    
    logits_pad = torch.nn.utils.rnn.pad_sequence(logits_batch, batch_first=True, padding_value=0)
    targets_pad = torch.nn.utils.rnn.pad_sequence(targets_batch, batch_first=True, padding_value=-1)
    
    input_lengths_tensor = torch.tensor(input_lengths_batch, dtype=torch.long)
    target_lengths_tensor = torch.tensor(target_lengths_batch, dtype=torch.long)
    
    return logits_pad, targets_pad, input_lengths_tensor, target_lengths_tensor


dataset = CustomDataset('ctc_fake_data.csv') 
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

for i, (logits, targets, input_lengths, target_lengths) in enumerate(dataloader):
    print(f"Batch {i+1}:")
    print(f"Logits: {logits.shape}, Targets: {targets.shape}")
    print(f"Input lengths: {input_lengths}, Target lengths: {target_lengths}")
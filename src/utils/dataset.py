import torch
from torch.utils.data import Dataset
import pandas as pd

class NaFe_Dataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        input_cols = ['F275W_abs', 'F336W_abs', 'F435W_abs', 'F606W_abs', 'F814W_abs']
        output_col = 'Na/Fe'
        
        df = pd.read_csv(csv_path)
        
        self.outputs = df[output_col].values  
        self.inputs = df[input_cols].values
        
        self.outputs = torch.tensor(self.outputs, dtype=dtype).reshape(-1, 1)
        self.inputs = torch.tensor(self.inputs, dtype=dtype)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

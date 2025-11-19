import torch
from torch.utils.data import Dataset
import pandas as pd


class CSVDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        output_cols: int = 3,
        input_cols: int = 5,
        dtype: torch.dtype = torch.float32
    ):

        super().__init__()
        df = pd.read_csv(csv_path)
        
        self.outputs = df.iloc[:, :output_cols].values  
        self.inputs = df.iloc[:, output_cols:output_cols + input_cols].values
        
        self.outputs = torch.tensor(self.outputs, dtype=dtype)
        self.inputs = torch.tensor(self.inputs, dtype=dtype)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
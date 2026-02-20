import torch
from torch.utils.data import Dataset
import pandas as pd


class NaFe_Dataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # original magnitude columns
        mag_cols = ['F275W_abs', 'F336W_abs', 'F435W_abs', 'F606W_abs', 'F814W_abs', 'Fe/H', 'age_Kruijssen']
        output_col = 'Na/Fe'

        df = pd.read_csv(csv_path)

        # construct the 7 input features:
        # F606W, F606W - F275W, F606W - F336W, F606W - F435W, F606W - F814W, Fe/H, age_Kruijssen
        f606 = df['F606W_abs']

        df_inputs = pd.DataFrame({
            'F606W': f606,
            'F606W_minus_F275W': f606 - df['F275W_abs'],
            'F606W_minus_F336W': f606 - df['F336W_abs'],
            'F606W_minus_F435W': f606 - df['F435W_abs'],
            'F606W_minus_F814W': f606 - df['F814W_abs'],
            'Fe/H': df['Fe/H'],
            'age_Kruijssen': df['age_Kruijssen']
        })

        # normalize each column: (x - mean) / std
        self.input_mean = df_inputs.mean()
        self.input_std = df_inputs.std().replace(0, 1.0)  
        df_inputs = (df_inputs - self.input_mean) / self.input_std

        self.inputs = torch.tensor(df_inputs.values, dtype=dtype)

        self.outputs = torch.tensor(
            df[output_col].values,
            dtype=dtype
        ).reshape(-1, 1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

class OFe_Dataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # original magnitude columns
        mag_cols = ['F275W_abs', 'F336W_abs', 'F435W_abs', 'F606W_abs', 'F814W_abs', 'Fe/H', 'age_Kruijssen']
        output_col = 'O/Fe'

        df = pd.read_csv(csv_path)

        # construct the 7 input features:
        # F606W, F606W - F275W, F606W - F336W, F606W - F435W, F606W - F814W, Fe/H, age_Kruijssen
        f606 = df['F606W_abs']

        df_inputs = pd.DataFrame({
            'F606W': f606,
            'F606W_minus_F275W': f606 - df['F275W_abs'],
            'F606W_minus_F336W': f606 - df['F336W_abs'],
            'F606W_minus_F435W': f606 - df['F435W_abs'],
            'F606W_minus_F814W': f606 - df['F814W_abs'],
            'Fe/H': df['Fe/H'],
            'age_Kruijssen': df['age_Kruijssen']
        })

        # normalize each column: (x - mean) / std
        self.input_mean = df_inputs.mean()
        self.input_std = df_inputs.std().replace(0, 1.0)  
        df_inputs = (df_inputs - self.input_mean) / self.input_std

        self.inputs = torch.tensor(df_inputs.values, dtype=dtype)

        self.outputs = torch.tensor(
            df[output_col].values,
            dtype=dtype
        ).reshape(-1, 1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
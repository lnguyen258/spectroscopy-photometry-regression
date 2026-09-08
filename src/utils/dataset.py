import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class NaFe_Dataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        dtype: torch.dtype = torch.float32,
        input_mean=None,
        input_std=None,
    ):
        super(NaFe_Dataset, self).__init__()

        # Input/Output columns
        mag_cols = ['F275W_abs', 'F336W_abs', 'F438W_abs', 'F606W_abs', 'F814W_abs', 'Fe/H', 'age_Kruijssen']
        output_col = 'Na/Fe'

        df = pd.read_csv(csv_path)

        # Define photometric ranges (adjust these as needed)
        ranges = {
            'F275W_abs': (2, 10),
            'F336W_abs': (0, 6),
            'F438W_abs': (-2, 5),
            'F606W_abs': (-5, 4),
            'F814W_abs': (-5, 3)
        }

        # Drop rows with any missing values in the 7 input columns
        df = df.dropna(subset=mag_cols + [output_col])

        # Drop rows outside photometric ranges
        for col, (min_val, max_val) in ranges.items():
            df = df[(df[col] >= min_val) & (df[col] <= max_val)]

        # Construct the 7 input features:
        f606 = df['F606W_abs']
        df_inputs = pd.DataFrame({
            'F606W': f606,
            'F606W_minus_F275W': f606 - df['F275W_abs'],
            'F606W_minus_F336W': f606 - df['F336W_abs'],
            'F606W_minus_F438W': f606 - df['F438W_abs'],
            'F606W_minus_F814W': f606 - df['F814W_abs'],
            'Fe/H': df['Fe/H'],
            'age_Kruijssen': df['age_Kruijssen']
        })

        # Use provided stats (e.g. from train set) or compute from this dataset
        if input_mean is not None and input_std is not None:
            self.input_mean = input_mean
            self.input_std = input_std
        else:
            self.input_mean = df_inputs.mean()
            self.input_std = df_inputs.std().replace(0, 1.0)

        df_inputs = (df_inputs - self.input_mean) / self.input_std

        self.inputs = torch.tensor(df_inputs.values, dtype=dtype)
        self.outputs = torch.tensor(
            df[output_col].values,
            dtype=dtype
        ).reshape(-1, 1)

    def get_sample_weights(self, n_bins: int = 10) -> torch.Tensor:
        targets = self.outputs.squeeze().numpy()
        bin_edges = np.linspace(targets.min(), targets.max(), n_bins + 1)
        bin_indices = np.digitize(targets, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(float)
        bin_counts[bin_counts == 0] = 1.0
        weights = 1.0 / np.sqrt(bin_counts[bin_indices])
        weights = weights / weights.sum() * len(weights)
        return torch.tensor(weights, dtype=torch.float)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class NaFe_Dataset_Colors(Dataset):
    """Uses 4 sequential colors + Fe/H + age as features (6 total).

    Colors: F275W-F336W, F336W-F438W, F438W-F606W, F606W-F814W
    """
    def __init__(
        self,
        csv_path: str,
        dtype: torch.dtype = torch.float32,
        input_mean=None,
        input_std=None,
    ):
        super(NaFe_Dataset_Colors, self).__init__()

        mag_cols = ['F275W_abs', 'F336W_abs', 'F438W_abs', 'F606W_abs', 'F814W_abs', 'Fe/H', 'age_Kruijssen']
        output_col = 'Na/Fe'

        df = pd.read_csv(csv_path)

        ranges = {
            'F275W_abs': (2, 10),
            'F336W_abs': (0, 6),
            'F438W_abs': (-2, 5),
            'F606W_abs': (-5, 4),
            'F814W_abs': (-5, 3)
        }

        df = df.dropna(subset=mag_cols + [output_col])

        for col, (min_val, max_val) in ranges.items():
            df = df[(df[col] >= min_val) & (df[col] <= max_val)]

        # Construct 6 features: 4 sequential colors + Fe/H + age
        df_inputs = pd.DataFrame({
            'F275W_minus_F336W': df['F275W_abs'] - df['F336W_abs'],
            'F336W_minus_F438W': df['F336W_abs'] - df['F438W_abs'],
            'F438W_minus_F606W': df['F438W_abs'] - df['F606W_abs'],
            'F606W_minus_F814W': df['F606W_abs'] - df['F814W_abs'],
            'Fe/H': df['Fe/H'],
            'age_Kruijssen': df['age_Kruijssen']
        })

        if input_mean is not None and input_std is not None:
            self.input_mean = input_mean
            self.input_std = input_std
        else:
            self.input_mean = df_inputs.mean()
            self.input_std = df_inputs.std().replace(0, 1.0)

        df_inputs = (df_inputs - self.input_mean) / self.input_std

        self.inputs = torch.tensor(df_inputs.values, dtype=dtype)
        self.outputs = torch.tensor(
            df[output_col].values,
            dtype=dtype
        ).reshape(-1, 1)

    def get_sample_weights(self, n_bins: int = 10) -> torch.Tensor:
        targets = self.outputs.squeeze().numpy()
        bin_edges = np.linspace(targets.min(), targets.max(), n_bins + 1)
        bin_indices = np.digitize(targets, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(float)
        bin_counts[bin_counts == 0] = 1.0
        weights = 1.0 / np.sqrt(bin_counts[bin_indices])
        weights = weights / weights.sum() * len(weights)
        return torch.tensor(weights, dtype=torch.float)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

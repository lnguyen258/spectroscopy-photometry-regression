import os
import yaml
import argparse

from src.config import TrainConfig, SineKAN_Config
from src.utils import CSVDataset
from src.models import SineKAN
from src.utils import plot_history
from trainer import Trainer

from torch.utils.data import random_split
import torch


parser = argparse.ArgumentParser(description="Train a KAN model for regression")

parser.add_argument('--config_path', type=str, default='./config/train_kan.yaml')
parser.add_argument('--data_dir', type=str, default='./data/merged_dropna.csv')

def main(
        config_path: str,
        data_dir: str,
):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize device
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize config
    model_config = SineKAN_Config.from_dict(config['model'])
    train_config = TrainConfig.from_dict(config['train'])

    # Initialize dataset
    dataset = CSVDataset(data_dir)
    val_ratio = 0.1
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Initialize model
    model = SineKAN(config=model_config)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        config=train_config,
    )

    # Train
    history, model = trainer.train()

    # Save the training history plot
    output_path = os.path.join(trainer.outputs_dir, f"{trainer.run_name}.png") if train_config.save_fig else None
    plot_history(history, save_fig=output_path)

if __name__ == "__main__":

    # Parse args
    args = parser.parse_args()

    main(
        config_path=args.config_path,
        data_dir=args.data_dir,
    )


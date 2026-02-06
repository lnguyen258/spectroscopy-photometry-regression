import os
import yaml
import argparse
import mlflow

from src.config import TrainConfig, SineKAN_Config
from src.utils import NaFe_Dataset
from src.models import SineKAN
from src.utils import plot_history
from trainer import Trainer

from torch.utils.data import random_split
import torch


parser = argparse.ArgumentParser(description="Train a KAN model for regression")

parser.add_argument('--config_path', type=str, default='config/train_kan.yaml')
parser.add_argument('--data_dir', type=str, default='data/Photometry+NaFe.csv')

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
    dataset = NaFe_Dataset(data_dir)
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

    # Initialize MLflow
    mlflow.set_experiment("SineKAN_Regression_Experiment")

    with mlflow.start_run(run_name=trainer.run_name):

        # Log configs and hyperparams for mlflow
        mlflow.log_params(config['model'])
        mlflow.log_params(config['train'])
        mlflow.log_param("device", str(device))
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("val_size", val_size)
    
        # Train
        history, model = trainer.train()

        # Inference
        metrics = trainer.inference()
        
        # Log metrics from history for mlflow
        for i in range(len(history['epoch'])):
            epoch = history['epoch'][i]
            mlflow.log_metric("train_loss", history['train_loss'][i], step=epoch)
            mlflow.log_metric("val_loss", history['val_loss'][i], step=epoch)

        # Save the training history plot & log plot for mlflow
        if train_config.save_fig:
            output_path = os.path.join(trainer.outputs_dir, f"{trainer.run_name}.png") 
            plot_history(history, save_fig=output_path)
            mlflow.log_artifact(output_path)

        # Log final model for mlflow
        mlflow.pytorch.log_model(model, "model")

        # Log best model for mlflow
        if train_config.save_best:
            mlflow.log_artifact(trainer.best_model_path)

if __name__ == "__main__":

    # Parse args
    args = parser.parse_args()

    main(
        config_path=args.config_path,
        data_dir=args.data_dir,
    )


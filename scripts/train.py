import os
import yaml
import argparse
import mlflow

from src.config import TrainConfig, SineKAN_Config
from src.utils import NaFe_Dataset, NaFe_Dataset_Colors
from src.models import MultiLayerSineKAN

DATASET_REGISTRY = {
    'default': NaFe_Dataset,
    'colors': NaFe_Dataset_Colors,
}
from src.utils import plot_history
from trainer import Trainer

from torch.utils.data import random_split, WeightedRandomSampler
import torch


parser = argparse.ArgumentParser(description="Train a KAN model for regression")

parser.add_argument('--config_path', type=str, default='config/train_kan_colors.yaml')
parser.add_argument('--train_data', type=str, default='data/Na_Fe_training_data.csv')
parser.add_argument('--test_data', type=str, default='data/Na_Fe_TEST_DATA.csv')
parser.add_argument('--dataset', type=str, default='colors', choices=['default', 'colors'])

def main(
        config_path: str,
        train_data: str,
        test_data: str,
        dataset: str,
):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize device
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize config
    model_config = SineKAN_Config.from_dict(config['model'])
    train_config = TrainConfig.from_dict(config['train'])

    # Initialize datasets — test set uses train normalization stats
    DatasetClass = DATASET_REGISTRY[dataset]
    full_train_dataset = DatasetClass(train_data)
    val_ratio = 0.1
    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    all_weights = full_train_dataset.get_sample_weights()
    train_weights = all_weights[train_dataset.indices]
    train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_dataset), replacement=True)

    test_dataset = DatasetClass(
        test_data,
        input_mean=full_train_dataset.input_mean,
        input_std=full_train_dataset.input_std,
    )

    # Initialize model
    model = MultiLayerSineKAN(config=model_config)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
        config=train_config,
        train_sampler=train_sampler,
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
        mlflow.log_param("test_size", len(test_dataset))

        # Train
        history, model = trainer.train()

        # Inference on test set
        test_loss, _, _ = trainer.inference(save_plot=train_config.save_fig)
        mlflow.log_metric("test_loss", test_loss)

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
            inference_plot_path = os.path.join(trainer.outputs_dir, f"{trainer.run_name}_inference.png")
            mlflow.log_artifact(inference_plot_path)

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
        train_data=args.train_data,
        test_data=args.test_data,
        dataset=args.dataset,
    )


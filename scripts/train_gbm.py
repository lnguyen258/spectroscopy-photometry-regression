import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

parser = argparse.ArgumentParser(description="Train a Gradient Boosting model for Na/Fe regression")
parser.add_argument('--train_data', type=str, default='data/Na_Fe_training_data.csv')
parser.add_argument('--test_data', type=str, default='data/Na_Fe_TEST_DATA.csv')
parser.add_argument('--dataset', type=str, default='colors', choices=['default', 'colors'])
parser.add_argument('--n_estimators', type=int, default=500)
parser.add_argument('--max_depth', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=0.05)
parser.add_argument('--subsample', type=float, default=0.8)
parser.add_argument('--colsample_bytree', type=float, default=0.8)
parser.add_argument('--min_child_weight', type=int, default=3)
parser.add_argument('--output_dir', type=str, default='log/GBM')


MAG_COLS = ['F275W_abs', 'F336W_abs', 'F438W_abs', 'F606W_abs', 'F814W_abs', 'Fe/H', 'age_Kruijssen']
OUTPUT_COL = 'Na/Fe'
RANGES = {
    'F275W_abs': (2, 10),
    'F336W_abs': (0, 6),
    'F438W_abs': (-2, 5),
    'F606W_abs': (-5, 4),
    'F814W_abs': (-5, 3),
}


def load_data(csv_path: str, dataset: str, input_mean=None, input_std=None):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=MAG_COLS + [OUTPUT_COL])
    for col, (lo, hi) in RANGES.items():
        df = df[(df[col] >= lo) & (df[col] <= hi)]

    if dataset == 'colors':
        features = pd.DataFrame({
            'F275W_minus_F336W': df['F275W_abs'] - df['F336W_abs'],
            'F336W_minus_F438W': df['F336W_abs'] - df['F438W_abs'],
            'F438W_minus_F606W': df['F438W_abs'] - df['F606W_abs'],
            'F606W_minus_F814W': df['F606W_abs'] - df['F814W_abs'],
            'Fe/H': df['Fe/H'],
            'age_Kruijssen': df['age_Kruijssen'],
        })
    else:
        f606 = df['F606W_abs']
        features = pd.DataFrame({
            'F606W': f606,
            'F606W_minus_F275W': f606 - df['F275W_abs'],
            'F606W_minus_F336W': f606 - df['F336W_abs'],
            'F606W_minus_F438W': f606 - df['F438W_abs'],
            'F606W_minus_F814W': f606 - df['F814W_abs'],
            'Fe/H': df['Fe/H'],
            'age_Kruijssen': df['age_Kruijssen'],
        })

    if input_mean is None:
        input_mean = features.mean()
        input_std = features.std().replace(0, 1.0)

    features = (features - input_mean) / input_std

    X = features.values
    y = df[OUTPUT_COL].values
    return X, y, input_mean, input_std


def get_sample_weights(y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    bin_edges = np.linspace(y.min(), y.max(), n_bins + 1)
    bin_indices = np.digitize(y, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(float)
    bin_counts[bin_counts == 0] = 1.0
    weights = 1.0 / np.sqrt(bin_counts[bin_indices])
    weights = weights / weights.sum() * len(weights)
    return weights


def main(args):
    os.makedirs(os.path.join(args.output_dir, 'output'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'best'), exist_ok=True)

    X_train, y_train, input_mean, input_std = load_data(args.train_data, args.dataset)
    X_test, y_test, _, _ = load_data(args.test_data, args.dataset, input_mean, input_std)
    sample_weights = get_sample_weights(y_train)

    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        objective='reg:squarederror',
        eval_metric='rmse',
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_mse = mean_squared_error(y_train, train_preds)
    test_mse = mean_squared_error(y_test, test_preds)
    print(f"\ntrain_loss (MSE): {train_mse:.4f} | test_loss (MSE): {test_mse:.4f}")
    print(f"Best iteration: {model.best_iteration}")

    feature_names = ['F275W-F336W', 'F336W-F438W', 'F438W-F606W', 'F606W-F814W', 'Fe/H', 'age'] \
        if args.dataset == 'colors' else \
        ['F606W', 'F606W-F275W', 'F606W-F336W', 'F606W-F438W', 'F606W-F814W', 'Fe/H', 'age']
    importances = model.feature_importances_
    print("\nFeature importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    _, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, test_preds, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Real Na/Fe', fontsize=12)
    ax.set_ylabel('Predicted Na/Fe', fontsize=12)
    ax.set_title(f'Na/Fe Prediction — GBM (MSE={test_mse:.4f})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, 'output', f'gbm_{args.dataset}_inference.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.show()

    model_path = os.path.join(args.output_dir, 'best', f'gbm_{args.dataset}.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

parser = argparse.ArgumentParser(description="Train GBM with outlier filtering for Na/Fe regression")
parser.add_argument('--train_data', type=str, default='data/Na_Fe_training_data.csv')
parser.add_argument('--test_data', type=str, default='data/Na_Fe_TEST_DATA.csv')
parser.add_argument('--dataset', type=str, default='colors', choices=['default', 'colors'])
parser.add_argument('--n_estimators', type=int, default=500)
parser.add_argument('--max_depth', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=0.05)
parser.add_argument('--subsample', type=float, default=0.8)
parser.add_argument('--colsample_bytree', type=float, default=0.8)
parser.add_argument('--min_child_weight', type=int, default=3)
parser.add_argument('--max_nafe', type=float, default=None,
                    help='Hard cap on Na/Fe — samples above this are excluded from both sets')
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


def remove_outliers(df: pd.DataFrame, col: str, max_val: float = None) -> pd.DataFrame:
    # IQR-based outlier removal
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (df[col] >= lower) & (df[col] <= upper)
    # Hard cap override
    if max_val is not None:
        mask = mask & (df[col] <= max_val)
    removed = (~mask).sum()
    if removed:
        print(f"  Removed {removed} outlier(s) — {col} outside [{lower:.3f}, {upper:.3f}]"
              + (f" and <= {max_val}" if max_val else ""))
    return df[mask]


def load_data(csv_path: str, dataset: str, max_nafe: float = None,
              input_mean=None, input_std=None, cluster_map: dict = None):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=MAG_COLS + [OUTPUT_COL])
    for col, (lo, hi) in RANGES.items():
        df = df[(df[col] >= lo) & (df[col] <= hi)]

    df = remove_outliers(df, OUTPUT_COL, max_val=max_nafe)

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

    # Add cluster ID — fit mapping on train, reuse on test; unseen clusters get -1
    if cluster_map is None:
        unique_clusters = sorted(df['NGC'].dropna().unique())
        cluster_map = {c: i for i, c in enumerate(unique_clusters)}
        print(f"  Clusters found: {list(cluster_map.keys())}")
    features['cluster_id'] = df['NGC'].map(cluster_map).fillna(-1).astype(int)

    return features.values, df[OUTPUT_COL].values, input_mean, input_std, cluster_map


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

    print("Loading train set:")
    X_train, y_train, input_mean, input_std, cluster_map = load_data(
        args.train_data, args.dataset, args.max_nafe)
    print("Loading test set:")
    X_test, y_test, _, _, _ = load_data(
        args.test_data, args.dataset, args.max_nafe, input_mean, input_std, cluster_map)

    sample_weights = get_sample_weights(y_train)
    print(f"\nTrain: {len(X_train)} samples | Test: {len(X_test)} samples")

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

    feature_names = ['F275W-F336W', 'F336W-F438W', 'F438W-F606W', 'F606W-F814W', 'Fe/H', 'age', 'cluster_id'] \
        if args.dataset == 'colors' else \
        ['F606W', 'F606W-F275W', 'F606W-F336W', 'F606W-F438W', 'F606W-F814W', 'Fe/H', 'age', 'cluster_id']
    importances = model.feature_importances_
    print("\nFeature importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    # Save predictions and targets for interpretation
    arrays_path = os.path.join(args.output_dir, 'output', f'gbm_clean_{args.dataset}_results.npz')
    np.savez(arrays_path, y_true=y_test, y_pred=test_preds, residuals=y_test - test_preds)
    print(f"Arrays saved to: {arrays_path}")

    residuals = y_test - test_preds
    print(f"\nResidual stats:")
    print(f"  Mean:   {residuals.mean():.4f}")
    print(f"  Std:    {residuals.std():.4f}")
    print(f"  Max over-pred:  {residuals.min():.4f}")
    print(f"  Max under-pred: {residuals.max():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Scatter: real vs predicted
    axes[0].scatter(y_test, test_preds, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    lim = [min(y_test.min(), test_preds.min()), max(y_test.max(), test_preds.max())]
    axes[0].plot(lim, lim, 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Real Na/Fe', fontsize=12)
    axes[0].set_ylabel('Predicted Na/Fe', fontsize=12)
    axes[0].set_title(f'Real vs Predicted (MSE={test_mse:.4f})', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Residual plot: real vs residual
    axes[1].scatter(y_test, residuals, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    axes[1].axhline(0, color='r', linestyle='--', lw=2, label='Zero residual')
    axes[1].set_xlabel('Real Na/Fe', fontsize=12)
    axes[1].set_ylabel('Residual (Real - Predicted)', fontsize=12)
    axes[1].set_title('Residuals vs Real Na/Fe', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('GBM clean — Test Set Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, 'output', f'gbm_clean_{args.dataset}_inference.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.show()

    model_path = os.path.join(args.output_dir, 'best', f'gbm_clean_{args.dataset}.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

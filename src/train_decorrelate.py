import datetime

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from sklearn.model_selection import KFold
import torch

from src.train import get_features
from src.data import get_dataframes, get_projects_info
from src.features import (
    add_github_projects_data,
    add_target_encoding,
    extract_activity_features,
    extract_ratio_features,
    extract_temporal_features,
)


def create_diversity_objective(base_model: lgb.Booster, alpha=0.1):
    """
    Creates a custom objective function for LightGBM that encourages functional diversity
    from a base model while maintaining accuracy.
    
    Args:
        base_model: Trained LightGBM model (lgb.Booster)
        alpha: Weight for diversity term (higher means more emphasis on being different)
    
    Returns:
        objective: Custom objective function compatible with LightGBM
    """
    def diversity_objective(y_pred, dataset: lgb.Dataset):
        """
        Custom objective expecting raw prediction scores.
        
        Args:
            y_pred: Raw predictions for current iteration
            dataset: lgb.Dataset object containing training data
        
        Returns:
            grad: First order gradients
            hess: Second order gradients
        """
        y_true = dataset.get_label()
        data = dataset.get_data()
        base_pred = base_model.predict(data)
        
        # Convert to torch tensors
        y_pred_t = torch.tensor(y_pred, requires_grad=True, dtype=torch.float32)
        y_true_t = torch.tensor(y_true, dtype=torch.float32)
        base_pred_t = torch.tensor(base_pred, dtype=torch.float32)
        
        # Calculate errors
        new_errors = y_pred_t - y_true_t
        base_errors = base_pred_t - y_true_t
        
        # Standardize errors
        new_errors_std = (new_errors - new_errors.mean()) / (new_errors.std() + 1e-8)
        base_errors_std = (base_errors - base_errors.mean()) / (base_errors.std() + 1e-8)
        
        # Loss = MSE - alpha * error_correlation
        mse = torch.mean((y_pred_t - y_true_t)**2)
        error_corr = torch.mean(new_errors_std * base_errors_std)
        loss = mse + alpha * error_corr
        
        # Get gradients
        loss.backward()
        grad = y_pred_t.grad.numpy()
        
        # Approximate hessian using finite differences
        epsilon = 1e-3
        y_pred_plus = y_pred + epsilon
        y_pred_plus_t = torch.tensor(y_pred_plus, requires_grad=True, dtype=torch.float32)
        
        new_errors_plus = y_pred_plus_t - y_true_t
        new_errors_std_plus = (new_errors_plus - new_errors_plus.mean()) / (new_errors_plus.std() + 1e-8)
        
        mse_plus = torch.mean((y_pred_plus_t - y_true_t)**2)
        error_corr_plus = torch.mean(new_errors_std_plus * base_errors_std)
        loss_plus = mse_plus - alpha * error_corr_plus
        
        loss_plus.backward()
        grad_plus = y_pred_plus_t.grad.numpy()
        
        hess = (grad_plus - grad) / epsilon
        
        return grad, hess
    
    return diversity_objective


def create_diversity_metric(base_model):
    """Creates a custom eval metric to track prediction diversity"""
    def diversity_metric(preds, train_data: lgb.Dataset):
        y_true = train_data.get_label()
        data = train_data.get_data()
        base_preds = base_model.predict(data)
        
        # Calculate error correlation
        error_corr = np.corrcoef(preds - y_true, base_preds - y_true)[0,1]
        print(f"Current error correlation: {error_corr:.4f}")  # Debug print

        mse = np.mean((preds - y_true)**2)
        print(f"Current MSE: {mse:.4f}")  # Debug print
        
        # Return (name, value, is_higher_better)
        # return 'error_corr', error_corr, False
        return mse, error_corr

    return diversity_metric


def main():
    # Load and prepare data
    print('Fetching train/test dataframes ...')
    df_train, df_test = get_dataframes()

    # Mirror training data
    df_train = pl.concat(
        [
            df_train,
            df_train.select(
                "id",
                pl.col("project_b").alias("project_a"),
                pl.col("project_a").alias("project_b"),
                pl.col("weight_b").alias("weight_a"),
                pl.col("weight_a").alias("weight_b"),
            ),
        ]
    )

    # Get unique projects and their info
    projects = (
        pl.concat(
            [
                df_train.get_column("project_a"),
                df_train.get_column("project_b"),
                df_test.get_column("project_a"),
                df_test.get_column("project_b"),
            ]
        )
        .unique()
        .to_list()
    )
    print('Getting project info ...')
    df_projects = get_projects_info(projects)

    # Add features
    print('Adding features ...')
    print('Adding github projects ...')
    print(df_train.shape)
    df_train_full = add_github_projects_data(df_train, df_projects)
    print(df_train_full.shape)
    print('Extracting temporal features ...')
    df_train_full = extract_temporal_features(df_train_full)
    print('Extracting activity features ...')
    df_train_full = extract_activity_features(df_train_full)
    # print('Adding target encoding ...')
    # df_train_full = add_target_encoding(df_train_full)
    print('Extracting ratio features ...')
    df_train_full = extract_ratio_features(df_train_full)

    print(df_test.shape)
    df_test_full = add_github_projects_data(df_test, df_projects)
    print(df_test_full.shape)
    df_test_full = extract_temporal_features(df_test_full)
    print(df_test_full.shape)
    df_test_full = extract_activity_features(df_test_full)
    print(df_test_full.shape)
    # df_test_full = add_target_encoding(df_test_full, df_train)
    # print(df_test_full.shape)
    df_test_full = extract_ratio_features(df_test_full)
    print(df_test_full.shape)

    # Prepare features and target
    features = get_features()
    X = df_train_full.select(features).to_numpy()
    y = df_train_full.get_column("weight_a").to_numpy()

    # Select important features
    with open('data/models/selected_features.txt', 'r') as f:
        selected_features = f.read().split(',')
    X = df_train_full.select(selected_features).to_numpy()

    # Form dataset
    train_data = lgb.Dataset(X, label=y, free_raw_data=False)

    # Load best hyperparameters
    import yaml
    with open('data/models/best_hyperparameters.yaml', 'r') as f:
        best_params = yaml.safe_load(f)

    # Load best model
    best_model = lgb.Booster(model_file = 'data/models/best_model.txt')

    # Train decorrelated model
    from collections import defaultdict
    stats = defaultdict(list)
    for alpha in np.linspace(0, 0.001, 11):
        diverse_obj = create_diversity_objective(best_model, alpha=alpha)
        diversity_metric = create_diversity_metric(best_model)

        import copy
        params_diverse = copy.copy(best_params)
        params_diverse['objective'] = diverse_obj
        params_diverse['metric'] = ['l2']
        # params_diverse['verbose'] = 1

        diverse_model = lgb.train(
            params_diverse, 
            train_data,
            feval=diversity_metric,
            callbacks=[
                lgb.log_evaluation(period=1),
            ],
            num_boost_round=100
        )

        mse, error_corr = diversity_metric(diverse_model.predict(X), train_data)

        stats['alpha'].append(alpha)
        stats['mse'].append(mse)
        stats['error_corr'].append(error_corr)

        # diverse_model.save_model(f'/data/models/model-alpha_{alpha}.txt')

        # Generate predictions
        X_test = df_test_full.select(selected_features).to_numpy()
        test_predictions = diverse_model.predict(X_test)
        test_predictions = pl.Series(test_predictions).round(6).clip(0)

        # Save submission
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df_test.select(pl.col("id"), test_predictions.alias("pred")).write_csv(
            f"data/submissions/submission_{timestamp}-alpha_{alpha}-mse_{mse:.4f}-corr_{error_corr:.2f}.csv"
        )

    print(stats)
    import code; code.interact(local=dict(globals(), **locals()))

    return


if __name__ == '__main__':
    main()
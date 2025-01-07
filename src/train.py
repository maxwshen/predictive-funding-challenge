import datetime

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from sklearn.model_selection import KFold

from src.data import get_dataframes, get_projects_info
from src.features import (
    add_github_projects_data,
    add_target_encoding,
    extract_activity_features,
    extract_ratio_features,
    extract_temporal_features,
)


def get_features() -> list[str]:
    """Get the list of features to use for training."""
    return [
        "is_private",
        "has_homepage",
        "size",
        "stars",
        "watchers",
        "has_projects",
        "has_pages",
        "has_wiki",
        "has_discussions",
        "forks",
        "is_archived",
        "is_disabled",
        "open_issues",
        "subscribers_count",
        "is_private_b",
        "has_homepage_b",
        "size_b",
        "stars_b",
        "watchers_b",
        "has_projects_b",
        "has_pages_b",
        "has_wiki_b",
        "has_discussions_b",
        "forks_b",
        "is_archived_b",
        "is_disabled_b",
        "open_issues_b",
        "subscribers_count_b",
        "age_days",
        "age_days_b",
        "days_since_update",
        "days_since_update_b",
        # Temporal decay features
        "stars_decay_0.0001",
        "stars_decay_b_0.0001",
        "forks_decay_0.0001",
        "forks_decay_b_0.0001",
        "issues_decay_0.0001",
        "issues_decay_b_0.0001",
        "stars_decay_0.001",
        "stars_decay_b_0.001",
        "forks_decay_0.001",
        "forks_decay_b_0.001",
        "issues_decay_0.001",
        "issues_decay_b_0.001",
        "stars_decay_0.01",
        "stars_decay_b_0.01",
        "forks_decay_0.01",
        "forks_decay_b_0.01",
        "issues_decay_0.01",
        "issues_decay_b_0.01",
        # Target encoding features
        "project_mean_weight_a",
        "project_mean_weight_b",
        # Ratio and interaction features
        "stars_ratio",
        "watchers_ratio",
        "forks_ratio",
        "size_ratio",
        "stars_forks_interaction",
        "stars_forks_interaction_b",
        "engagement_score",
        "engagement_score_b",
        "stars_per_day",
        "stars_per_day_b",
        "forks_per_day",
        "forks_per_day_b",
        "log_stars",
        "log_stars_b",
        "log_watchers",
        "log_watchers_b",
        "log_forks",
        "log_forks_b",
    ]


def get_base_model_params() -> dict:
    """Get the base LightGBM model parameters that won't be tuned."""
    return {
        "objective": "regression",
        "metric": "mse",
        "force_col_wise": True,
        "verbose": -1,
    }


def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        X: Feature matrix
        y: Target values

    Returns:
        Mean cross-validation MSE score
    """
    param = {
        **get_base_model_params(),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(param, train_data, valid_sets=[val_data])
        y_pred = model.predict(X_val)
        mse = float(np.mean((y_val - y_pred) ** 2))
        cv_scores.append(mse)

    return float(np.mean(cv_scores))


def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> dict:
    """
    Optimize LightGBM hyperparameters using Optuna.

    Args:
        X: Feature matrix
        y: Target values
        n_trials: Number of optimization trials

    Returns:
        Dictionary of best hyperparameters
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print("\nBest hyperparameters:")
    for param, value in study.best_params.items():
        print(f"{param}: {value}")
    print(f"Best MSE: {study.best_value:.4f}")

    return {**get_base_model_params(), **study.best_params}


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    features: list[str],
    importance_threshold: float = 0.01,
) -> list[str]:
    """
    Select features based on importance scores from a preliminary model.

    Args:
        X: Feature matrix
        y: Target values
        features: List of feature names
        importance_threshold: Minimum importance score to keep a feature

    Returns:
        List of selected feature names
    """
    # Train a preliminary model
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(get_base_model_params(), train_data)

    # Get feature importance scores
    importance_scores = model.feature_importance(importance_type="gain")
    total_importance = importance_scores.sum()

    # Select features with importance above threshold
    selected_features = [
        feature
        for feature, importance in zip(features, importance_scores)
        if importance / total_importance > importance_threshold
    ]

    print(f"\nSelected {len(selected_features)} features out of {len(features)}")
    print("\nTop 10 features by importance:")
    sorted_features = sorted(
        zip(features, importance_scores), key=lambda x: x[1], reverse=True
    )
    for feature, importance in sorted_features[:10]:
        print(f"{feature}: {importance / total_importance:.4f}")

    return selected_features


def train_and_evaluate(
    X: np.ndarray, y: np.ndarray, params: dict | None = None
) -> tuple[float, float]:
    """
    Train and evaluate the model using cross-validation.

    Args:
        X: Feature matrix
        y: Target values
        params: Model parameters. If None, use base parameters.

    Returns:
        Tuple of mean MSE and standard deviation
    """
    if params is None:
        params = get_base_model_params()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(params, train_data, valid_sets=[val_data])

        y_pred = model.predict(X_val)
        mse = float(np.mean((y_val - y_pred) ** 2))
        cv_scores.append(mse)

    cv_scores = np.array(cv_scores)
    return float(cv_scores.mean()), float(cv_scores.std())


def train_final_model(
    X: np.ndarray, y: np.ndarray, params: dict | None = None
) -> lgb.Booster:
    """
    Train the final model on all data.

    Args:
        X: Feature matrix
        y: Target values
        params: Model parameters. If None, use base parameters.

    Returns:
        Trained LightGBM model
    """
    if params is None:
        params = get_base_model_params()

    train_data = lgb.Dataset(X, label=y)
    return lgb.train(params, train_data)


def main():
    # Load and prepare data
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
    df_projects = get_projects_info(projects)

    # Add features
    df_train_full = add_github_projects_data(df_train, df_projects)
    df_train_full = extract_temporal_features(df_train_full)
    df_train_full = extract_activity_features(df_train_full)
    df_train_full = add_target_encoding(df_train_full)
    df_train_full = extract_ratio_features(df_train_full)

    df_test_full = add_github_projects_data(df_test, df_projects)
    df_test_full = extract_temporal_features(df_test_full)
    df_test_full = extract_activity_features(df_test_full)
    df_test_full = add_target_encoding(df_test_full, df_train)
    df_test_full = extract_ratio_features(df_test_full)

    # Prepare features and target
    features = get_features()
    X = df_train_full.select(features).to_numpy()
    y = df_train_full.get_column("weight_a").to_numpy()

    # Select important features
    selected_features = select_features(X, y, features)
    X = df_train_full.select(selected_features).to_numpy()

    # Optimize hyperparameters
    print("\nOptimizing hyperparameters...")
    best_params = optimize_hyperparameters(X, y)

    # Train and evaluate with best parameters
    mean_mse, std_mse = train_and_evaluate(X, y, best_params)
    print(
        f"\nCross-validation MSE with optimized parameters: {mean_mse:.4f} (+/- {std_mse:.4f})"
    )

    # Train final model with best parameters
    model = train_final_model(X, y, best_params)

    # Generate predictions
    X_test = df_test_full.select(selected_features).to_numpy()
    test_predictions = model.predict(X_test)
    test_predictions = pl.Series(test_predictions).round(6).clip(0)

    # Save submission
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df_test.select(pl.col("id"), test_predictions.alias("pred")).write_csv(
        f"data/submissions/submission_{timestamp}-mse_{mean_mse:.6f}.csv"
    )


if __name__ == "__main__":
    main()

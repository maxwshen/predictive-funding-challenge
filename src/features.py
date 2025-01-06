from datetime import datetime

import numpy as np
import polars as pl


def mirror_train_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Mirror the training data to create a symmetric dataset.
    """
    return pl.concat(
        [
            df,
            df.select(
                "id",
                pl.col("project_b").alias("project_a"),
                pl.col("project_a").alias("project_b"),
                pl.col("weight_b").alias("weight_a"),
                pl.col("weight_a").alias("weight_b"),
            ),
        ]
    )


def add_github_projects_data(
    df: pl.DataFrame, df_projects: pl.DataFrame
) -> pl.DataFrame:
    """
    Add GitHub projects data to both projects in the DataFrame.
    """

    df_projects = df_projects.select(
        pl.col("full_name").str.to_lowercase().alias("project_id"),
        pl.col("private").alias("is_private"),
        pl.col("description"),
        pl.col("created_at"),
        pl.col("updated_at"),
        pl.col("homepage").is_not_null().alias("has_homepage"),
        pl.col("size"),
        pl.col("stargazers_count").alias("stars"),
        pl.col("watchers_count").alias("watchers"),
        pl.col("language"),
        pl.col("has_projects"),
        pl.col("has_pages"),
        pl.col("has_wiki"),
        pl.col("has_discussions"),
        pl.col("forks_count").alias("forks"),
        pl.col("archived").alias("is_archived"),
        pl.col("disabled").alias("is_disabled"),
        pl.col("open_issues_count").alias("open_issues"),
        pl.col("subscribers_count"),
    )

    df = df.join(
        df_projects,
        left_on="project_a",
        right_on="project_id",
        how="left",
        suffix="_a",
    )

    df = df.join(
        df_projects,
        left_on="project_b",
        right_on="project_id",
        how="left",
        suffix="_b",
    )

    return df


def extract_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract temporal features from repository data.

    Args:
        df: Input DataFrame with repository data

    Returns:
        DataFrame with added temporal features
    """
    features = df.clone()

    if "created_at" in features.columns and "updated_at" in features.columns:
        features = features.with_columns(
            [
                pl.col("created_at")
                .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ")
                .alias("created_dt"),
                pl.col("updated_at")
                .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ")
                .alias("updated_dt"),
                pl.col("created_at_b")
                .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ")
                .alias("created_dt_b"),
                pl.col("updated_at_b")
                .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ")
                .alias("updated_dt_b"),
            ]
        )

        # Calculate days since last update
        now = pl.lit(datetime.now())
        features = features.with_columns(
            [
                ((now - pl.col("updated_dt")).dt.total_days()).alias(
                    "days_since_update"
                ),
                ((now - pl.col("updated_dt_b")).dt.total_days()).alias(
                    "days_since_update_b"
                ),
                (
                    (
                        pl.col("updated_dt").cast(pl.Int64)
                        - pl.col("created_dt").cast(pl.Int64)
                    )
                    / (24 * 3600)
                ).alias("age_days"),
                (
                    (
                        pl.col("updated_dt_b").cast(pl.Int64)
                        - pl.col("created_dt_b").cast(pl.Int64)
                    )
                    / (24 * 3600)
                ).alias("age_days_b"),
            ]
        )

    return features


def extract_activity_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract activity-based features from repository data.

    Args:
        df: Input DataFrame with repository data

    Returns:
        DataFrame with added activity features
    """
    features = df.clone()

    # Add temporal decay features
    features = features.with_columns(
        [
            (
                pl.col("stars")
                * pl.col("days_since_update").map_elements(
                    lambda x: np.exp(-0.001 * x), return_dtype=pl.Float64
                )
            ).alias("stars_decay"),
            (
                pl.col("stars_b")
                * pl.col("days_since_update_b").map_elements(
                    lambda x: np.exp(-0.001 * x), return_dtype=pl.Float64
                )
            ).alias("stars_decay_b"),
            (
                pl.col("forks")
                * pl.col("days_since_update").map_elements(
                    lambda x: np.exp(-0.001 * x), return_dtype=pl.Float64
                )
            ).alias("forks_decay"),
            (
                pl.col("forks_b")
                * pl.col("days_since_update_b").map_elements(
                    lambda x: np.exp(-0.001 * x), return_dtype=pl.Float64
                )
            ).alias("forks_decay_b"),
        ]
    )

    # Add interaction features
    features = features.with_columns(
        [
            (pl.col("stars").fill_null(0) * pl.col("open_issues").fill_null(0)).alias(
                "stars_issues_interaction"
            ),
            (
                pl.col("stars_b").fill_null(0) * pl.col("open_issues_b").fill_null(0)
            ).alias("stars_issues_interaction_b"),
            (pl.col("forks").fill_null(0) * pl.col("open_issues").fill_null(0)).alias(
                "forks_issues_interaction"
            ),
            (
                pl.col("forks_b").fill_null(0) * pl.col("open_issues_b").fill_null(0)
            ).alias("forks_issues_interaction_b"),
        ]
    )

    return features


def extract_ratio_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract ratio-based features from repository data.

    Args:
        df: Input DataFrame with repository data

    Returns:
        DataFrame with added ratio features
    """
    features = df.clone()

    # Basic ratios
    features = features.with_columns(
        [
            (pl.col("stars") / (pl.col("stars") + pl.col("stars_b"))).alias(
                "stars_ratio"
            ),
            (pl.col("watchers") / (pl.col("watchers") + pl.col("watchers_b"))).alias(
                "watchers_ratio"
            ),
            (pl.col("forks") / (pl.col("forks") + pl.col("forks_b"))).alias(
                "forks_ratio"
            ),
            (pl.col("size") / (pl.col("size") + pl.col("size_b"))).alias("size_ratio"),
            (
                pl.col("open_issues")
                / (pl.col("open_issues") + pl.col("open_issues_b"))
            ).alias("issues_ratio"),
            (
                pl.col("subscribers_count")
                / (pl.col("subscribers_count") + pl.col("subscribers_count_b"))
            ).alias("subscribers_count_ratio"),
        ]
    )

    # Log transforms
    features = features.with_columns(
        [
            pl.col("stars").fill_null(0).log1p().alias("log_stars"),
            pl.col("stars_b").fill_null(0).log1p().alias("log_stars_b"),
            pl.col("watchers").fill_null(0).log1p().alias("log_watchers"),
            pl.col("watchers_b").fill_null(0).log1p().alias("log_watchers_b"),
            pl.col("forks").fill_null(0).log1p().alias("log_forks"),
            pl.col("forks_b").fill_null(0).log1p().alias("log_forks_b"),
        ]
    )

    return features

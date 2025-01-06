import os
from pathlib import Path
from typing import Dict, List, Tuple

import httpx
import polars as pl


def get_dataframes() -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Fetch the dataset and test data from GitHub and return as Polars DataFrames.
    Uses local cache if available.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: Tuple containing the train and test DataFrames
    """
    raw_dir = Path("../data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_cache = raw_dir / "dataset.csv"
    test_cache = raw_dir / "test.csv"

    repository_url = (
        "https://raw.githubusercontent.com/deepfunding/mini-contest/refs/heads/main/"
    )

    # Try to load from cache first
    if train_cache.exists() and test_cache.exists():
        df_train = pl.read_csv(train_cache)
        df_test = pl.read_csv(test_cache)
    else:
        # Download and cache if not available
        df_train = pl.read_csv(f"{repository_url}/dataset.csv")
        df_test = pl.read_csv(f"{repository_url}/test.csv")

        # Cache the raw files
        df_train.write_csv(train_cache)
        df_test.write_csv(test_cache)

    # Light preprocessing to get project IDs instead of full URLs
    df_train = df_train.with_columns(
        pl.col("project_a").str.split("github.com/").list.last().alias("project_a"),
        pl.col("project_b").str.split("github.com/").list.last().alias("project_b"),
    )

    df_test = df_test.with_columns(
        pl.col("project_a").str.split("github.com/").list.last().alias("project_a"),
        pl.col("project_b").str.split("github.com/").list.last().alias("project_b"),
    )

    return df_train, df_test


def get_repository_info(repository_id: str, client: httpx.Client) -> Dict:
    """
    Fetch repository information from GitHub API for a given repo URL.

    Args:
        repo_url: GitHub repository URL
        client: httpx.Client instance to use for requests

    Returns:
        Dict containing repository information or empty dict if request fails
    """
    api_url = f"https://api.github.com/repos/{repository_id}"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        response = client.get(api_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        print(f"Error fetching data for {repository_id}")
        print(response.text)
        return {}


def get_projects_info(projects: List[str]) -> pl.DataFrame:
    """
    Fetch project information from GitHub API for a list of project IDs and return as a Polars DataFrame.
    Uses local cache if available.

    Args:
        projects: List of GitHub repository IDs

    Returns:
        pl.DataFrame containing GitHub project information for all projects
    """
    processed_dir = Path("../data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_file = processed_dir / "projects_info.parquet"

    # Try to load existing cache
    if cache_file.exists():
        cached_df = pl.read_parquet(cache_file)
        cached_projects = set(cached_df["full_name"].to_list())
        # Only fetch missing projects
        projects_to_fetch = [p for p in projects if p not in cached_projects]
        if not projects_to_fetch:
            return cached_df
    else:
        cached_df = None
        projects_to_fetch = projects

    data = []
    with httpx.Client(
        transport=httpx.HTTPTransport(retries=5, verify=False),
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    ) as client:
        for project_id in projects_to_fetch:
            info = get_repository_info(project_id, client)
            if info:
                data.append(info)

    new_df = pl.DataFrame(data)

    # Merge with cache if it exists
    if cached_df is not None:
        final_df = pl.concat([cached_df, new_df])
    else:
        final_df = new_df

    # Update cache
    final_df.write_parquet(cache_file)

    return final_df

import os
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import httpx
import polars as pl


def get_dataframes() -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Fetch the dataset and test data from GitHub and return as Polars DataFrames.
    Uses local cache if available.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: Tuple containing the train and test DataFrames
    """
    raw_dir = Path("data/raw")
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


import time
def get_repository_info(repository_id: str, client: httpx.Client) -> Dict:
    """
    Fetch repository information from GitHub API for a given repo ID.
    Includes rate limit handling and better error reporting.
    """
    api_url = f"https://api.github.com/repos/{repository_id}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        response = client.get(api_url, headers=headers)
        
        # Handle rate limiting
        if response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers:
            remaining = int(response.headers['X-RateLimit-Remaining'])
            if remaining == 0:
                reset_time = int(response.headers['X-RateLimit-Reset'])
                sleep_time = reset_time - time.time() + 1
                if sleep_time > 0:
                    print(f"Rate limit exceeded. Waiting {sleep_time:.0f} seconds...")
                    time.sleep(sleep_time)
                    return get_repository_info(repository_id, client)
        
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        print(f"Error fetching data for {repository_id}: {str(e)}")
        if response.status_code != 404:  # Don't print response text for 404s
            print(f"Response: {response.text}")
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
    print(f'Found {len(projects)} projects...')
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_file = processed_dir / "projects_info.parquet"

    if cache_file.exists():
        return pl.read_parquet(cache_file)

    data = []
    with httpx.Client(
        transport=httpx.HTTPTransport(retries=5, verify=False),
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    ) as client:
        for project_id in tqdm(projects):
            info = get_repository_info(project_id, client)
            if info:
                data.append(info)

    df = pl.DataFrame(data)
    df.write_parquet(cache_file)
    return df
import os
from typing import Dict, List, Tuple

import httpx
import polars as pl


def get_dataframes() -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Fetch the dataset and test data from GitHub and return as a Polars DataFrames.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: Tuple containing the train and test DataFrames
    """

    repository_url = (
        "https://raw.githubusercontent.com/deepfunding/mini-contest/refs/heads/main/"
    )

    df_train = pl.read_csv(f"{repository_url}/dataset.csv")
    df_test = pl.read_csv(f"{repository_url}/test.csv")

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

    Args:
        projects: List of GitHub repository IDs

    Returns:
        pl.DataFrame containing GitHub project information for all projects
    """
    data = []

    with httpx.Client(
        transport=httpx.HTTPTransport(retries=5, verify=False),
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    ) as client:
        for project_id in projects:
            info = get_repository_info(project_id, client)
            if info:
                data.append(info)

    return pl.DataFrame(data)

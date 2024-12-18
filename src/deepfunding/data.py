import os
from typing import Dict, List

import httpx
import polars as pl


def get_graph_dataframe() -> pl.DataFrame:
    """
    Fetch the dependency graph data from GitHub and return as a Polars DataFrame.

    Returns:
        pl.DataFrame: DataFrame containing the graph links with source and target columns
    """
    url = "https://raw.githubusercontent.com/deepfunding/dependency-graph/refs/heads/main/graph/unweighted_graph.json"

    with httpx.Client() as client:
        response = client.get(url)
        response.raise_for_status()
        graph_data: Dict = response.json()

    return pl.DataFrame(graph_data.get("links")).drop("weight")


def get_repository_info(repo_url: str, client: httpx.Client) -> Dict:
    """
    Fetch repository information from GitHub API for a given repo URL.

    Args:
        repo_url: GitHub repository URL
        client: httpx.Client instance to use for requests

    Returns:
        Dict containing repository information or empty dict if request fails
    """
    _, _, _, owner, repo = repo_url.rstrip("/").split("/")
    api_url = f"https://api.github.com/repos/{owner}/{repo}"

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
        return {}


def get_repository_info_dataframe(repo_urls: List[str]) -> pl.DataFrame:
    """
    Fetch repository information from GitHub API for a list of repo URLs and return as a Polars DataFrame.

    Args:
        repo_urls: List of GitHub repository URLs

    Returns:
        pl.DataFrame containing repository information for all repos
    """
    repos_data = []

    with httpx.Client(
        transport=httpx.HTTPTransport(retries=5, verify=False),
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    ) as client:
        for url in repo_urls:
            repo_info = get_repository_info(url, client)
            if repo_info:
                repos_data.append(repo_info)

    return pl.DataFrame(repos_data)

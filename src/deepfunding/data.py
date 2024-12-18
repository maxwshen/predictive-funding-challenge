from typing import Dict

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

import polars as pl


def check_weights_sum_to_one(
    df: pl.DataFrame, group_col: str, weight_col: str = "weight"
) -> bool:
    """
    Check if weights sum to 1.0 within each group in a Polars DataFrame.

    Args:
        df: Input Polars DataFrame
        group_col: Column name to group by
        weight_col: Column containing weights (default: "weight")

    Returns:
        bool: True if weights sum to 1.0 for each group, False otherwise
    """
    # Group by the specified column and sum weights
    sums = df.group_by(group_col).agg(pl.col(weight_col).sum())

    # Check if all sums are approximately 1.0 (using small epsilon for float comparison)
    return all((sums[weight_col] - 1.0).abs() < 1e-10)

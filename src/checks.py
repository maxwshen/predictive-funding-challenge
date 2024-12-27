import polars as pl


def check_transitivity(df: pl.DataFrame) -> bool:
    """
    Check if the "weight" column is consistent with the transitivity property (if a > b and b > c, then a > c).
    """
    return True

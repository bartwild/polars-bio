from typing import Union, Optional
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from .context import ctx
import pyarrow as pa


def visualize_base_content(
    df: Union[pl.DataFrame, pd.DataFrame],
    figsize: tuple = (12, 6),
    title: str = "Base Content Across Positions",
    xlabel: str = "Position",
    ylabel: str = "Percentage",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize base content percentages across positions.

    Parameters
    ----------
    df : Union[pl.DataFrame, pd.DataFrame]
        DataFrame with base content percentages
    figsize : tuple, optional
        Figure size, by default (12, 6)
    title : str, optional
        Plot title, by default "Base Content Across Positions"
    xlabel : str, optional
        X-axis label, by default "Position"
    ylabel : str, optional
        Y-axis label, by default "Percentage"
    save_path : Optional[str], optional
        Path to save the figure, by default None

    Returns
    -------
    None
        Displays the plot
    """
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    required_cols = ["position", "A", "C", "G", "T", "N"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    plt.figure(figsize=figsize)

    for base in ["A", "C", "G", "T", "N"]:
        plt.plot(df["position"], df[base], label=base)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def base_content(
    df: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame],
    output_type: str = "polars.DataFrame"
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Calculate base content percentages for each position in sequences using parallel processing.

    Parameters
    ----------
    df : Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame]
        DataFrame containing a 'sequence' column with DNA/RNA sequences
    output_type : str, optional
        Output type, by default "polars.DataFrame"

    Returns
    -------
    Union[pl.DataFrame, pd.DataFrame]
        DataFrame with base content percentages for each position

    Examples
    --------
    >>> import polars_bio as pb
    >>> df = pb.read_fastq("example.fastq").collect()
    >>> base_content_df = pb.qc.base_content(df)
    >>> pb.qc.visualize_base_content(base_content_df)
    """
    from polars_bio.polars_bio import base_content_analysis

    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    elif isinstance(df, pl.LazyFrame):
        df = df.collect()

    if "sequence" not in df.columns:
        raise ValueError("DataFrame must contain a 'sequence' column")
    result_df = base_content_analysis(ctx, df)

    collected_data = result_df.collect()
    if isinstance(collected_data, pl.DataFrame):
        polars_df = collected_data
    elif isinstance(collected_data, pa.Table):
        polars_df = pl.from_arrow(collected_data)
    elif isinstance(collected_data, list) and all(isinstance(batch, pa.RecordBatch) for batch in collected_data):
        arrow_table = pa.Table.from_batches(collected_data)
        polars_df = pl.from_arrow(arrow_table)

    input_type = type(df).__name__
    if output_type is None:
        if input_type == "DataFrame" and "pandas" in pd.DataFrame.__module__:
            output_type = "pandas.DataFrame"
        else:
            output_type = "polars.DataFrame"

    # Convert to requested output type
    if output_type == "pandas.DataFrame":
        return polars_df.to_pandas()
    else:
        return polars_df

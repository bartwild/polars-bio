from typing import Union, Optional
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from .context import ctx
import pyarrow as pa


def base_content(
    df: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame],
    output_type: str = "polars.DataFrame"
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Calculate base content percentages for each position in sequences.
    
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
    # Import here to avoid circular imports
    from polars_bio.polars_bio import base_content_analysis
    
    # Convert to polars DataFrame if needed
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    elif isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    # Check if 'sequence' column exists
    if "sequence" not in df.columns:
        raise ValueError("DataFrame must contain a 'sequence' column")
    
    # Calculate base content
    record_batches = base_content_analysis(ctx, df)
    
    # Convert the list of PyArrow RecordBatches to a Polars DataFrame
    if isinstance(record_batches, list) and len(record_batches) > 0:
        # Convert the list of record batches to a PyArrow Table
        arrow_table = pa.Table.from_batches(record_batches)
        # Convert the PyArrow Table to a Polars DataFrame
        result = pl.from_arrow(arrow_table)
    else:
        # If we got an empty result or not a list, create an empty DataFrame with the expected columns
        result = pl.DataFrame({
            "position": [],
            "A": [],
            "C": [],
            "G": [],
            "T": [],
            "N": []
        })
    
    # Convert to requested output type
    if output_type == "pandas.DataFrame":
        return result.to_pandas()
    else:
        return result


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
    # Convert to pandas if it's a polars DataFrame
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    
    required_cols = ["position", "A", "C", "G", "T", "N"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Plot each base
    for base in ["A", "C", "G", "T", "N"]:
        plt.plot(df["position"], df[base], label=base)
    
    # Add labels and legend
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # Show the plot
    plt.show()


def base_content_parallel(
    df: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame],
    num_threads: int = 4,
    output_type: str = "polars.DataFrame"
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Calculate base content percentages for each position in sequences using parallel processing.
    
    Parameters
    ----------
    df : Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame]
        DataFrame containing a 'sequence' column with DNA/RNA sequences
    num_threads : int, optional
        Number of threads to use for parallel processing, by default 4
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
    >>> base_content_df = pb.qc.base_content_parallel(df, num_threads=8)
    >>> pb.qc.visualize_base_content(base_content_df)
    """
    # Import here to avoid circular imports
    from polars_bio.polars_bio import base_content_analysis_parallel
    
    # Convert to polars DataFrame if needed
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    elif isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    # Check if 'sequence' column exists
    if "sequence" not in df.columns:
        raise ValueError("DataFrame must contain a 'sequence' column")
    
    # Calculate base content using parallel implementation
    record_batches = base_content_analysis_parallel(ctx, df, num_threads)
    
    # Convert the list of PyArrow RecordBatches to a Polars DataFrame
    if isinstance(record_batches, list) and len(record_batches) > 0:
        # Convert the list of record batches to a PyArrow Table
        arrow_table = pa.Table.from_batches(record_batches)
        # Convert the PyArrow Table to a Polars DataFrame
        result = pl.from_arrow(arrow_table)
    else:
        # If we got an empty result or not a list, create an empty DataFrame with the expected columns
        result = pl.DataFrame({
            "position": [],
            "A": [],
            "C": [],
            "G": [],
            "T": [],
            "N": []
        })
    
    # Convert to requested output type
    if output_type == "pandas.DataFrame":
        return result.to_pandas()
    else:
        return result
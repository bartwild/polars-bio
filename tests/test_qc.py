import polars_bio as pb
import polars as pl
import pandas as pd
from _expected import DATA_DIR
import matplotlib.pyplot as plt


class TestBaseContent:
    df = pb.read_fastq(f"{DATA_DIR}/io/fastq/test.fastq").collect()
    
    def test_base_content(self):
        # Test with polars DataFrame
        result = pb.qc.base_content(self.df)
        assert isinstance(result, pl.DataFrame)
        assert "position" in result.columns
        assert "A" in result.columns
        assert "C" in result.columns
        assert "G" in result.columns
        assert "T" in result.columns
        assert "N" in result.columns
        
        # Check that percentages sum to 100 for each position
        for row in result.iter_rows(named=True):
            total = row["A"] + row["C"] + row["G"] + row["T"] + row["N"]
            assert abs(total - 100.0) < 1e-10
        
        # Test with pandas DataFrame
        pdf = self.df.to_pandas()
        result_pd = pb.qc.base_content(pdf, output_type="pandas.DataFrame")
        assert isinstance(result_pd, pd.DataFrame)
        
        # Test with LazyFrame
        ldf = self.df.lazy()
        result_lazy = pb.qc.base_content(ldf)
        assert isinstance(result_lazy, pl.DataFrame)
    
    def test_visualization(self):
        # Just test that it runs without errors
        result = pb.qc.base_content(self.df)
        
        # Save the current state of plt.show
        original_show = plt.show
        
        # Replace plt.show with a no-op function
        plt.show = lambda: None
        
        try:
            pb.qc.visualize_base_content(result)
        finally:
            # Restore plt.show
            plt.show = original_show
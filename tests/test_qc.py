import polars_bio as pb
import polars as pl
import pandas as pd
from _expected import DATA_DIR
import matplotlib.pyplot as plt


class TestBaseContent:
    df = pb.read_fastq(f"{DATA_DIR}/io/fastq/test.fastq").collect()
    
    def test_base_content(self):
        result = pb.qc.base_content(self.df)
        assert isinstance(result, pl.DataFrame)
        assert "position" in result.columns
        assert "A" in result.columns
        assert "C" in result.columns
        assert "G" in result.columns
        assert "T" in result.columns
        assert "N" in result.columns
        
        for row in result.iter_rows(named=True):
            total = row["A"] + row["C"] + row["G"] + row["T"] + row["N"]
            assert abs(total - 100.0) < 1e-10
        pdf = self.df.to_pandas()
        result_pd = pb.qc.base_content(pdf, output_type="pandas.DataFrame")
        assert isinstance(result_pd, pd.DataFrame)
        
        ldf = self.df.lazy()
        result_lazy = pb.qc.base_content(ldf)
        assert isinstance(result_lazy, pl.DataFrame)
    
    def test_visualization(self):
        result = pb.qc.base_content(self.df)
        
        original_show = plt.show
        
        plt.show = lambda: None
        
        try:
            pb.qc.visualize_base_content(result)
        finally:
            plt.show = original_show
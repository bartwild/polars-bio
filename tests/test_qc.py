import polars_bio as pb
import polars as pl
import pandas as pd
from _expected import DATA_DIR
import pytest
import matplotlib.pyplot as plt
import random


class TestBaseContent:
    @classmethod
    def setup_class(cls):
        cls.df = pb.read_fastq(f"{DATA_DIR}/io/fastq/test.fastq").collect()

    def test_base_content_polars(self):
        result = pb.qc.base_content(self.df)
        assert isinstance(result, pl.DataFrame)
        assert set(["position", "A", "C", "G", "T", "N"]).issubset(result.columns)

        for row in result.iter_rows(named=True):
            total = row["A"] + row["C"] + row["G"] + row["T"] + row["N"]
            assert abs(total - 100.0) < 1e-10, f"Base % sum != 100: {total}"

    def test_base_content_pandas(self):
        pdf = self.df.to_pandas()
        result = pb.qc.base_content(pdf, output_type="pandas.DataFrame")

        if "base" in result.columns and isinstance(result["base"].iloc[0], dict):
            base_df = pd.json_normalize(result["base"])
            result = pd.concat([result.drop(columns=["base"]), base_df], axis=1)

        assert isinstance(result, pd.DataFrame)
        assert set(["position", "A", "C", "G", "T", "N"]).issubset(result.columns)

        for _, row in result.iterrows():
            total = row["A"] + row["C"] + row["G"] + row["T"] + row["N"]
            assert abs(total - 100.0) < 1e-10

    def test_base_content_lazyframe(self):
        ldf = self.df.lazy()
        result = pb.qc.base_content(ldf)
        assert isinstance(result, pl.DataFrame)
        assert set(["position", "A", "C", "G", "T", "N"]).issubset(result.columns)

    def test_visualization_runs_without_error(self):
        result = pb.qc.base_content(self.df)
        original_show = plt.show
        plt.show = lambda: None

        try:
            pb.qc.visualize_base_content(result)
        finally:
            plt.show = original_show

    def test_visualization_raises_on_missing_columns(self):
        invalid_df = pl.DataFrame({"position": [1, 2, 3], "A": [20, 30, 25]})
        with pytest.raises(ValueError, match="DataFrame must contain columns"):
            pb.qc.visualize_base_content(invalid_df)

    def test_invalid_input_type(self):
        with pytest.raises(ValueError, match="DataFrame must contain a 'sequence' column"):
            pb.qc.base_content(pl.DataFrame({"id": [1, 2], "name": ["a", "b"]}))


def generate_sequences(n: int, length: int, bases="ACGTN") -> list[str]:
    return [
        "".join(random.choices(bases, k=length))
        for _ in range(n)
    ]


class TestSyntheticSequences:
    def test_all_same_base(self):
        df = pl.DataFrame({"sequence": ["A" * 50] * 10})
        result = pb.qc.base_content(df)
        assert all(result["A"] == 100.0)
        for base in ["C", "G", "T", "N"]:
            assert all(result[base] == 0.0)

    def test_mixed_random_bases(self):
        seqs = generate_sequences(n=100, length=75)
        df = pl.DataFrame({"sequence": seqs})
        result = pb.qc.base_content(df)
        assert isinstance(result, pl.DataFrame)
        assert result["position"].max() == 74
        for base in ["A", "C", "G", "T", "N"]:
            assert base in result.columns

    def test_different_lengths_should_warn_or_return_empty(self):
        df = pl.DataFrame({"sequence": ["ACGT" * 10, "ACGT" * 9]})
        result = pb.qc.base_content(df)
        assert result.shape[0] == 40

    def test_empty_sequences(self):
        df = pl.DataFrame({"sequence": [""] * 5})
        result = pb.qc.base_content(df)
        assert result.shape[0] == 0

    def test_non_dna_characters(self):
        # Unexpected letters like 'X' or digits
        df = pl.DataFrame({"sequence": ["AXGT" * 10, "CC33TTAA" * 5]})
        result = pb.qc.base_content(df)
        assert isinstance(result, pl.DataFrame)
        assert all(col in result.columns for col in ["A", "C", "G", "T", "N"])

    def test_large_scale(self):
        # 10k sequences of length 100
        df = pl.DataFrame({"sequence": generate_sequences(10_000, 100)})
        result = pb.qc.base_content(df)
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 100
        for row in result.iter_rows(named=True):
            total = row["A"] + row["C"] + row["G"] + row["T"] + row["N"]
            assert abs(total - 100.0) < 1e-6


class TestBaseContentSQL:
    @classmethod
    def setup_class(cls):
        fastq_file = f"{DATA_DIR}/io/fastq/test.fastq"
        pb.read_fastq(fastq_file).collect()
        cls.df = pb.read_fastq(fastq_file).collect()
        pb.ctx.set_option("datafusion.execution.target_partitions", "2")

    def test_base_content_sql_result_shape_and_columns(self):
        result = pb.sql("SELECT base_content(sequence) AS base FROM test").collect().unnest("base")
        assert isinstance(result, pl.DataFrame)
        assert set(["position", "A", "C", "G", "T", "N"]).issubset(result.columns)

    def test_base_content_sql_sum_to_100(self):
        result = pb.sql("SELECT base_content(sequence) AS base FROM test").collect().unnest("base")
        for row in result.iter_rows(named=True):
            total = row["A"] + row["C"] + row["G"] + row["T"] + row["N"]
            assert abs(total - 100.0) < 1e-10, f"Base % sum != 100: {total}"

    def test_sql_query_on_subset(self):
        result = pb.sql("SELECT base_content(sequence) AS base FROM test WHERE LENGTH(sequence) >= 2").collect().unnest("base")
        assert result.shape[0] > 0
        assert set(["position", "A", "C", "G", "T", "N"]).issubset(result.columns)

    def test_sql_vs_api_equivalence(self):
        sql_result = pb.sql("SELECT base_content(sequence) AS base FROM test").collect().unnest("base")
        api_result = pb.qc.base_content(self.df)

        assert sql_result.shape == api_result.shape
        assert sql_result.columns == api_result.columns
        for col in ["A", "C", "G", "T", "N", "position"]:
            assert sql_result[col].equals(api_result[col]), f"Mismatch in column: {col}"

    def test_sql_with_alias_and_filter(self):
        query = """
            SELECT base_content(sequence) AS base_content_result
            FROM test
            WHERE LENGTH(sequence) >= 2
        """
        result = pb.sql(query).collect().unnest("base_content_result")
        assert result.shape[0] > 0
        assert "A" in result.columns
        assert result["position"].max() >= 0

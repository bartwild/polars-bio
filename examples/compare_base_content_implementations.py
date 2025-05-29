import polars as pl
import json
import matplotlib.pyplot as plt
import numpy as np


def load_format1(json_str):
    """
    Wczytuje dane z formatu 1: [{"position":0,"A":42.0,"C":51.0,"G":48.0,"T":46.0,"N":13.0}, ...]
    """
    data = json.loads(json_str)
    return pl.DataFrame(data)


def load_format2(json_str):
    """
    Wczytuje dane z formatu 2: {"values": [{"base": "A", "count": 42, "pos": 0}, ...]}
    """
    data = json.loads(json_str)
    df = pl.DataFrame(data["values"])

    pivot_df = df.pivot(
        index="pos",
        columns="base",
        values="count"
    ).sort("pos")

    # Zmiana nazwy kolumny indeksu na "position"
    pivot_df = pivot_df.rename({"pos": "position"})

    return pivot_df


def calculate_base_stats(df):
    """
    Oblicza statystyki dla każdej zasady (A, C, G, T, N)
    """
    bases = ["A", "C", "G", "T", "N"]
    stats = {}

    for base in bases:
        if base in df.columns:
            base_values = df[base].to_numpy()
            stats[base] = {
                "mean": np.mean(base_values),
                "median": np.median(base_values),
                "std": np.std(base_values),
                "min": np.min(base_values),
                "max": np.max(base_values),
                "sum": np.sum(base_values),
                "count": len(base_values)
            }

    return stats


def plot_base_averages(stats):
    """
    Tworzy wykres słupkowy średnich wartości dla każdej zasady
    """
    bases = ["A", "C", "G", "T", "N"]
    available_bases = [base for base in bases if base in stats]
    means = [stats[base]["mean"] for base in available_bases]

    if not means:
        print("Brak danych do wyświetlenia wykresu średnich wartości.")
        return

    plt.figure(figsize=(10, 6))
    bars = plt.bar(available_bases, means,
                   color=[{'A': 'green', 'C': 'blue', 'G': 'black', 'T': 'red', 'N': 'gray'}[base] for base in available_bases])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom')

    plt.title('Średnia zawartość zasad w sekwencji')
    plt.xlabel('Zasada')
    plt.ylabel('Średnia zawartość (%)')
    if means:
        plt.ylim(0, max(means) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_base_distribution(df):
    """
    Tworzy wykres liniowy pokazujący rozkład zasad na różnych pozycjach
    """
    plt.figure(figsize=(14, 8))

    plt.plot(df["position"], df["A"], '-', label='A', color='green', linewidth=2)
    plt.plot(df["position"], df["C"], '-', label='C', color='blue', linewidth=2)
    plt.plot(df["position"], df["G"], '-', label='G', color='black', linewidth=2)
    plt.plot(df["position"], df["T"], '-', label='T', color='red', linewidth=2)
    if "N" in df.columns:
        plt.plot(df["position"], df["N"], '-', label='N', color='gray', linewidth=2)

    plt.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Idealny rozkład (25%)')

    plt.title('Rozkład zasad na pozycjach')
    plt.xlabel('Pozycja w odczycie')
    plt.ylabel('Zawartość (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_gc_content(df):
    """
    Oblicza zawartość GC na każdej pozycji
    """
    gc_df = df.with_columns(
        ((pl.col("G") + pl.col("C")) / (pl.col("A") + pl.col("C") + pl.col("G") + pl.col("T")) * 100).alias("GC_content")
    )
    return gc_df


def plot_gc_content(gc_df):
    """
    Tworzy wykres liniowy pokazujący zawartość GC na różnych pozycjach
    """
    plt.figure(figsize=(14, 6))

    plt.plot(gc_df["position"], gc_df["GC_content"], '-', color='purple', linewidth=2)

    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Idealny rozkład GC (50%)')

    mean_gc = gc_df["GC_content"].mean()
    plt.axhline(y=mean_gc, color='red', linestyle='-', alpha=0.7,
                label=f'Średnia zawartość GC: {mean_gc:.2f}%')

    plt.title('Zawartość GC na pozycjach')
    plt.xlabel('Pozycja w odczycie')
    plt.ylabel('Zawartość GC (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()


def calculate_base_bias(df):
    """
    Oblicza nierównowagę zasad na każdej pozycji
    """
    expected = 25.0

    bias_df = df.with_columns([
        (pl.col("A") - expected).abs().alias("A_bias"),
        (pl.col("C") - expected).abs().alias("C_bias"),
        (pl.col("G") - expected).abs().alias("G_bias"),
        (pl.col("T") - expected).abs().alias("T_bias"),
    ])

    bias_df = bias_df.with_columns(
        (pl.col("A_bias") + pl.col("C_bias") + pl.col("G_bias") + pl.col("T_bias")).alias("total_bias")
    )

    return bias_df


def plot_base_bias(bias_df):
    """
    Tworzy wykres liniowy pokazujący nierównowagę zasad na różnych pozycjach
    """
    plt.figure(figsize=(14, 6))

    plt.plot(bias_df["position"], bias_df["total_bias"], '-', color='darkred', linewidth=2)

    mean_bias = bias_df["total_bias"].mean()
    plt.axhline(y=mean_bias, color='blue', linestyle='-', alpha=0.7,
                label=f'Średnia nierównowaga: {mean_bias:.2f}%')

    plt.title('Nierównowaga zasad na pozycjach')
    plt.xlabel('Pozycja w odczycie')
    plt.ylabel('Całkowita nierównowaga (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_base_content(json_str1, json_str2=None):
    """
    Analizuje dane zawartości zasad z jednego lub dwóch formatów JSON
    """
    df1 = load_format1(json_str1)
    print("Dane z formatu 1:")
    print(df1.head())

    stats1 = calculate_base_stats(df1)
    print("\nStatystyki dla formatu 1:")
    for base, base_stats in stats1.items():
        print(f"\n{base}:")
        for stat_name, stat_value in base_stats.items():
            print(f"  {stat_name}: {stat_value:.2f}")

    print("\nWizualizacja średnich wartości dla formatu 1:")
    plot_base_averages(stats1)

    print("\nWizualizacja rozkładu zasad dla formatu 1:")
    plot_base_distribution(df1)

    gc_df1 = calculate_gc_content(df1)
    print("\nWizualizacja zawartości GC dla formatu 1:")
    plot_gc_content(gc_df1)

    bias_df1 = calculate_base_bias(df1)
    print("\nWizualizacja nierównowagi zasad dla formatu 1:")
    plot_base_bias(bias_df1)

    if json_str2:
        df2 = load_format2(json_str2)
        print("\nDane z formatu 2:")
        print(df2.head())

        are_identical = df1.equals(df2)
        print(f"\nCzy dane są identyczne? {are_identical}")

        if not are_identical:
            print("\nAnalizowanie różnic między formatami...")

            common_cols = set(df1.columns).intersection(set(df2.columns))

            diff_stats = {}
            for col in common_cols:
                if col == "position":
                    continue

                df_diff = df1.select(pl.col("position"), pl.col(col)).join(
                    df2.select(pl.col("position"), pl.col(col).alias(f"{col}_2")),
                    on="position"
                )

                df_diff = df_diff.with_columns(
                    (pl.col(col) - pl.col(f"{col}_2")).alias("diff"),
                    ((pl.col(col) - pl.col(f"{col}_2")).abs() / pl.col(col) * 100).alias("percent_diff")
                )

                diff_values = df_diff["diff"].to_numpy()
                percent_diff_values = df_diff["percent_diff"].to_numpy()
                percent_diff_values = percent_diff_values[~np.isnan(percent_diff_values)]  # Usuń NaN

                diff_stats[col] = {
                    "mean_diff": np.mean(diff_values),
                    "median_diff": np.median(diff_values),
                    "max_diff": np.max(diff_values),
                    "min_diff": np.min(diff_values),
                    "std_diff": np.std(diff_values),
                    "mean_percent_diff": np.mean(percent_diff_values) if len(percent_diff_values) > 0 else 0,
                    "max_percent_diff": np.max(percent_diff_values) if len(percent_diff_values) > 0 else 0,
                }

                top_diff_positions = df_diff.sort("diff", descending=True).head(5)
                print(f"\nTop 5 pozycji z największymi różnicami dla {col}:")
                print(top_diff_positions)

            print("\nStatystyki różnic między formatami:")
            for col, stats in diff_stats.items():
                print(f"\n{col}:")
                for stat_name, stat_value in stats.items():
                    print(f"  {stat_name}: {stat_value:.4f}")

            print("\nWizualizacja porównawcza dla każdej zasady:")
            for base in ["A", "C", "G", "T", "N"]:
                if base in df1.columns and base in df2.columns:
                    plt.figure(figsize=(14, 6))

                    plt.plot(df1["position"], df1[base], '-', label=f'Format 1 - {base}', linewidth=2)
                    plt.plot(df2["position"], df2[base], '--', label=f'Format 2 - {base}', linewidth=2)

                    plt.title(f'Porównanie zawartości {base} między formatami')
                    plt.xlabel('Pozycja w odczycie')
                    plt.ylabel('Zawartość (%)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()

            print("\nWizualizacja różnic między formatami:")
            plt.figure(figsize=(14, 8))

            for base in ["A", "C", "G", "T", "N"]:
                if base in common_cols:
                    df_diff = df1.select(pl.col("position"), pl.col(base)).join(
                        df2.select(pl.col("position"), pl.col(base).alias(f"{base}_2")),
                        on="position"
                    ).with_columns(
                        (pl.col(base) - pl.col(f"{base}_2")).alias(f"{base}_diff")
                    )

                    colors = {'A': 'green', 'C': 'blue', 'G': 'black', 'T': 'red', 'N': 'gray'}
                    plt.plot(df_diff["position"], df_diff[f"{base}_diff"], '-',
                             label=f'Różnica {base}', color=colors[base], linewidth=1.5)

            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.title('Różnice w zawartości zasad między formatami')
            plt.xlabel('Pozycja w odczycie')
            plt.ylabel('Różnica (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            gc_df2 = calculate_gc_content(df2)

            plt.figure(figsize=(14, 6))
            plt.plot(gc_df1["position"], gc_df1["GC_content"], '-', label='Format 1 GC', color='purple', linewidth=2)
            plt.plot(gc_df2["position"], gc_df2["GC_content"], '--', label='Format 2 GC', color='orange', linewidth=2)

            plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Idealny rozkład GC (50%)')

            mean_gc1 = gc_df1["GC_content"].mean()
            mean_gc2 = gc_df2["GC_content"].mean()
            plt.axhline(y=mean_gc1, color='red', linestyle='-', alpha=0.7,
                        label=f'Średnia GC Format 1: {mean_gc1:.2f}%')
            plt.axhline(y=mean_gc2, color='blue', linestyle='-', alpha=0.7,
                        label=f'Średnia GC Format 2: {mean_gc2:.2f}%')

            plt.title('Porównanie zawartości GC między formatami')
            plt.xlabel('Pozycja w odczycie')
            plt.ylabel('Zawartość GC (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 100)
            plt.tight_layout()
            plt.show()

            bias_df2 = calculate_base_bias(df2)

            plt.figure(figsize=(14, 6))
            plt.plot(bias_df1["position"], bias_df1["total_bias"], '-',
                     label='Format 1 nierównowaga', color='darkred', linewidth=2)
            plt.plot(bias_df2["position"], bias_df2["total_bias"], '--',
                     label='Format 2 nierównowaga', color='darkblue', linewidth=2)

            mean_bias1 = bias_df1["total_bias"].mean()
            mean_bias2 = bias_df2["total_bias"].mean()
            plt.axhline(y=mean_bias1, color='red', linestyle='-', alpha=0.7,
                        label=f'Średnia nierównowaga Format 1: {mean_bias1:.2f}%')
            plt.axhline(y=mean_bias2, color='blue', linestyle='-', alpha=0.7,
                        label=f'Średnia nierównowaga Format 2: {mean_bias2:.2f}%')

            plt.title('Porównanie nierównowagi zasad między formatami')
            plt.xlabel('Pozycja w odczycie')
            plt.ylabel('Całkowita nierównowaga (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            comparison_results = {
                "format1_stats": {base: {k: float(v) for k, v in stats.items()} for base, stats in stats1.items()},
                "format2_stats": calculate_base_stats(df2),
                "differences": {col: {k: float(v) for k, v in stats.items()} for col, stats in diff_stats.items()},
                "gc_content": {
                    "format1_mean": float(mean_gc1),
                    "format2_mean": float(mean_gc2),
                    "difference": float(mean_gc1 - mean_gc2)
                },
                "base_bias": {
                    "format1_mean": float(mean_bias1),
                    "format2_mean": float(mean_bias2),
                    "difference": float(mean_bias1 - mean_bias2)
                }
            }

            with open("base_content_comparison_results.json", "w") as f:
                json.dump(comparison_results, f, indent=4)

            print("\nWyniki porównania zostały zapisane do pliku 'base_content_comparison_results.json'")

        else:
            print("\nOba formaty zawierają identyczne dane. Nie ma potrzeby dalszej analizy.")

    return


if __name__ == "__main__":

    with open("examples/data2.json", 'r') as f:
        json_str1 = f.read()

    json_str2 = None
    with open("examples/data.json", 'r') as f:
        json_str2 = f.read()

    analyze_base_content(json_str1, json_str2)

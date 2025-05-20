import os
import json
import time
from rich import print
from rich.table import Table
from rich.box import MARKDOWN
import subprocess
import tempfile
from pathlib import Path

os.makedirs("results", exist_ok=True)

BENCH_DATA_ROOT = os.getenv("BENCH_DATA_ROOT", "/home/rafalunix/polars-bio/tests/data/small3.fastq")

if not BENCH_DATA_ROOT or not Path(BENCH_DATA_ROOT).exists():
    raise ValueError(f"BENCH_DATA_ROOT is not set or file does not exist: {BENCH_DATA_ROOT}")

NUM_REPEATS = 100
THREADS_LIST = [8, 4, 1]

TEST_CASES = [{
    "file_path": BENCH_DATA_ROOT,
    "name": "large",
    "description": "Large FASTQ file (~1GB)"
}]

def run_single_benchmark(test_case, threads, repeats):
    result_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    result_file.close()

    script_content = f"""
import time
import polars as pl
import polars_bio as pb
import json

file_path = r"{test_case['file_path']}"
num_threads = {threads}
num_repeats = {repeats}

io_times = []
compute_times = []
total_times = []

for _ in range(num_repeats):
    # Force garbage collection between runs
    import gc
    gc.collect()

    start_total = time.time()

    # Simulated I/O timer
    start_io = time.time()
    df = pb.read_fastq(file_path)
    collected = df.collect()
    io_time = time.time() - start_io

    # Compute timer
    start_compute = time.time()
    seq = collected.select("sequence")
    result = pb.qc.base_content(seq, num_threads=num_threads)
    compute_time = time.time() - start_compute

    total_time = time.time() - start_total

    io_times.append(io_time)
    compute_times.append(compute_time)
    total_times.append(total_time)

results = {{
    "threads": num_threads,
    "io": {{
        "min": min(io_times),
        "max": max(io_times),
        "mean": sum(io_times) / len(io_times)
    }},
    "compute": {{
        "min": min(compute_times),
        "max": max(compute_times),
        "mean": sum(compute_times) / len(compute_times)
    }},
    "total": {{
        "min": min(total_times),
        "max": max(total_times),
        "mean": sum(total_times) / len(total_times)
    }}
}}

with open(r"{result_file.name}", "w") as f:
    json.dump(results, f)
"""

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as f:
        f.write(script_content)
        script_path = f.name

    try:
        completed = subprocess.run(
            ["python", script_path],
            capture_output=True,
            check=True,
            text=True
        )
        if completed.stdout.strip():
            print("[green]Subprocess stdout:[/green]")
            print(completed.stdout)
        if completed.stderr.strip():
            print("[red]Subprocess stderr:[/red]")
            print(completed.stderr)

        with open(result_file.name, "r") as f:
            results = json.load(f)

        return results

    except subprocess.CalledProcessError as e:
        print(f"[red]Benchmark failed for {threads} threads:[/red] {e.stderr}")
        return None
    finally:
        os.unlink(script_path)
        os.unlink(result_file.name)

def create_speedup_table(results, baseline_idx, title, metric):
    table = Table(title=title, box=MARKDOWN)
    table.add_column("Threads", justify="right", style="cyan")
    table.add_column(f"{metric} Time (s)", justify="right", style="green")
    table.add_column(f"{metric} Speedup", justify="right", style="magenta")

    baseline = results[baseline_idx]["mean"]
    for result in results:
        speedup = baseline / result["mean"] if result["mean"] > 0 else 0
        table.add_row(
            str(result["threads"]),
            f"{result['mean']:.6f}",
            f"{speedup:.2f}x"
        )
    return table

def main():
    for test_case in TEST_CASES:
        print(f"[bold]Running benchmark: {test_case['name']} ({test_case['description']})[/bold]")

        all_results = {
            "test_case": test_case["name"],
            "description": test_case["description"],
            "io_results": [],
            "compute_results": [],
            "total_results": []
        }

        raw_results = []
        for threads in THREADS_LIST:
            print(f"[bold]Running thread number: {threads}[/bold]")
            result = run_single_benchmark(test_case, threads, NUM_REPEATS)
            if result:
                raw_results.append(result)

        raw_results.sort(key=lambda x: x["threads"])

        for result in raw_results:
            all_results["io_results"].append({**result["io"], "threads": result["threads"]})
            all_results["compute_results"].append({**result["compute"], "threads": result["threads"]})
            all_results["total_results"].append({**result["total"], "threads": result["threads"]})

        if all_results["io_results"]:
            print(create_speedup_table(all_results["io_results"], 0, f"I/O Speedup - {test_case['name']}", "I/O"))
        if all_results["compute_results"]:
            print(create_speedup_table(all_results["compute_results"], 0, f"Compute Speedup - {test_case['name']}", "Compute"))
        if all_results["total_results"]:
            print(create_speedup_table(all_results["total_results"], 0, f"Total Speedup - {test_case['name']}", "Total"))

        result_path = os.path.join("results", f"base_content_{test_case['name']}_results.json")
        with open(result_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[green]Results saved to:[/green] {result_path}")

if __name__ == "__main__":
    main()

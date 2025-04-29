import os
import timeit
import numpy as np
import pandas as pd
import polars as pl
from rich import print
from rich.box import MARKDOWN
from rich.table import Table
import json
import polars_bio as pb
import time  # Add this import for the time.time() function

# Set environment variables
os.environ["POLARS_MAX_THREADS"] = "1"  # For single-threaded tests
BENCH_DATA_ROOT = os.getenv("BENCH_DATA_ROOT", '/home/dbartosiak/praca/polars-bio/tests/data/qc/example.fastq')

if BENCH_DATA_ROOT is None:
    raise ValueError("BENCH_DATA_ROOT is not set")

# Test parameters
num_repeats = 3
num_executions = 3
test_threads = [1, 2, 4, 8]  # For parallel tests

# Test cases - adjust paths as needed
test_cases = [{
        "file_path": f"{BENCH_DATA_ROOT}",
        "name": "large",
        "description": "Large FASTQ file (~1GB)"
    }
]

# Define functions to benchmark
def polars_bio_base_content(file_path):
    """Benchmark polars-bio base_content function"""
    df = pb.read_fastq(file_path)
    result = pb.qc.base_content(df.collect())
    return result

def polars_bio_base_content_parallel(file_path, threads):
    """Benchmark polars-bio base_content function with parallel execution"""
    # Set the number of threads for DataFusion
    pb.ctx.set_option("datafusion.execution.target_partitions", str(threads))
    
    # Also set these options to ensure proper parallelization
    pb.ctx.set_option("datafusion.optimizer.repartition_joins", "true")
    pb.ctx.set_option("datafusion.optimizer.repartition_file_scans", "true")
    pb.ctx.set_option("datafusion.execution.coalesce_batches", "false")
    
    # Ensure Polars is also using the right number of threads
    # Use the correct method for your version of Polars
    try:
        # For newer versions of Polars
        pl.Config.set_global_threads(threads)
    except AttributeError:
        try:
            # For older versions of Polars
            pl.set_global_threads(threads)
        except AttributeError:
            # If neither method works, try environment variable
            os.environ["POLARS_MAX_THREADS"] = str(threads)
            print(f"Set POLARS_MAX_THREADS environment variable to {threads}")
    
    df = pb.read_fastq(file_path)
    result = pb.qc.base_content(df.collect())
    return result

def polars_bio_base_content_parallel_separated(file_path, threads):
    """Benchmark with I/O and computation separated"""
    # Set thread options
    pb.ctx.set_option("datafusion.execution.target_partitions", str(threads))
    pb.ctx.set_option("datafusion.optimizer.repartition_joins", "true")
    pb.ctx.set_option("datafusion.optimizer.repartition_file_scans", "true")
    pb.ctx.set_option("datafusion.execution.coalesce_batches", "false")
    
    try:
        # For newer versions of Polars
        pl.Config.set_global_threads(threads)
    except AttributeError:
        try:
            # For older versions of Polars
            pl.set_global_threads(threads)
        except AttributeError:
            # If neither method works, try environment variable
            os.environ["POLARS_MAX_THREADS"] = str(threads)
    
    # First, read the file (time this separately)
    start_io = time.time()
    df = pb.read_fastq(file_path).collect()
    io_time = time.time() - start_io
    
    # Then, process the data (time this separately)
    start_compute = time.time()
    result = pb.qc.base_content(df)
    compute_time = time.time() - start_compute
    
    total_time = io_time + compute_time
    
    print(f"I/O time: {io_time:.6f}s, Compute time: {compute_time:.6f}s, Total: {total_time:.6f}s")
    
    return result, io_time, compute_time, total_time

# Create results directory
os.makedirs("results", exist_ok=True)

# Run single-threaded benchmarks
print("Running single-threaded benchmarks...")
for t in test_cases:
    results = []
    
    print(f"Testing {t['name']} ({t['description']})...")
    
    # Benchmark polars-bio base_content
    try:
        times = timeit.repeat(
            lambda: polars_bio_base_content(t["file_path"]),
            repeat=num_repeats,
            number=num_executions
        )
        
        per_run_times = [time / num_executions for time in times]
        results.append({
            "name": "polars_bio_base_content",
            "min": min(per_run_times),
            "max": max(per_run_times),
            "mean": np.mean(per_run_times)
        })
    except Exception as e:
        print(f"Error benchmarking polars_bio_base_content: {e}")
    
    # Create Rich table
    table = Table(title=f"Base Content Benchmark Results - {t['name']}", box=MARKDOWN)
    table.add_column("Function", justify="left", style="cyan", no_wrap=True)
    table.add_column("Min (s)", justify="right", style="green")
    table.add_column("Max (s)", justify="right", style="green")
    table.add_column("Mean (s)", justify="right", style="green")
    
    # Add rows to the table
    for result in results:
        table.add_row(
            result["name"],
            f"{result['min']:.6f}",
            f"{result['max']:.6f}",
            f"{result['mean']:.6f}"
        )
    
    # Display the table
    print(table)
    
    # Save results to JSON
    benchmark_results = {
        "test_case": t["name"],
        "description": t["description"],
        "results": results
    }
    json.dump(benchmark_results, open(f"results/base_content_{t['name']}.json", "w"))

# Run parallel benchmarks
print("\nRunning parallel benchmarks...")
for t in test_cases:
    for threads in test_threads:
        results = []
        
        print(f"Testing {t['name']} with {threads} threads...")
        
        # Benchmark polars-bio base_content with parallel execution
        try:
            times = timeit.repeat(
                lambda: polars_bio_base_content_parallel(t["file_path"], threads),
                repeat=num_repeats,
                number=num_executions
            )
            
            per_run_times = [time / num_executions for time in times]
            results.append({
                "name": f"polars_bio_base_content_{threads}_threads",
                "min": min(per_run_times),
                "max": max(per_run_times),
                "mean": np.mean(per_run_times),
                "threads": threads
            })
        except Exception as e:
            print(f"Error benchmarking polars_bio_base_content with {threads} threads: {e}")
        
        # Create Rich table
        table = Table(title=f"Base Content Benchmark Results - {t['name']} ({threads} threads)", box=MARKDOWN)
        table.add_column("Function", justify="left", style="cyan", no_wrap=True)
        table.add_column("Min (s)", justify="right", style="green")
        table.add_column("Max (s)", justify="right", style="green")
        table.add_column("Mean (s)", justify="right", style="green")
        table.add_column("Threads", justify="right", style="magenta")
        
        # Add rows to the table
        for result in results:
            table.add_row(
                result["name"],
                f"{result['min']:.6f}",
                f"{result['max']:.6f}",
                f"{result['mean']:.6f}",
                str(result["threads"])
            )
        
        # Display the table
        print(table)
        
        # Save results to JSON
        benchmark_results = {
            "test_case": t["name"],
            "description": t["description"],
            "threads": threads,
            "results": results
        }
        json.dump(benchmark_results, open(f"results/base_content_{t['name']}_{threads}_threads.json", "w"))

# Compare speedup across different thread counts
print("\nComparing speedup across thread counts...")
for t in test_cases:
    speedup_results = []
    
    # Load single-threaded results as baseline
    try:
        with open(f"results/base_content_{t['name']}.json", "r") as f:
            baseline_data = json.load(f)
            baseline_time = baseline_data["results"][0]["mean"]
            
            speedup_results.append({
                "threads": 1,
                "time": baseline_time,
                "speedup": 1.0
            })
            
            # Calculate speedup for each thread count
            for threads in test_threads[1:]:  # Skip the first one (1 thread)
                try:
                    with open(f"results/base_content_{t['name']}_{threads}_threads.json", "r") as f:
                        thread_data = json.load(f)
                        thread_time = thread_data["results"][0]["mean"]
                        speedup = baseline_time / thread_time
                        
                        speedup_results.append({
                            "threads": threads,
                            "time": thread_time,
                            "speedup": speedup
                        })
                except Exception as e:
                    print(f"Error loading results for {threads} threads: {e}")
            
            # Create Rich table for speedup comparison
            table = Table(title=f"Speedup Comparison - {t['name']}", box=MARKDOWN)
            table.add_column("Threads", justify="right", style="cyan")
            table.add_column("Time (s)", justify="right", style="green")
            table.add_column("Speedup", justify="right", style="magenta")
            
            # Add rows to the table
            for result in speedup_results:
                table.add_row(
                    str(result["threads"]),
                    f"{result['time']:.6f}",
                    f"{result['speedup']:.2f}x"
                )
            
            # Display the table
            print(table)
            
            # Save speedup results to JSON
            json.dump({
                "test_case": t["name"],
                "description": t["description"],
                "speedup_results": speedup_results
            }, open(f"results/base_content_{t['name']}_speedup.json", "w"))
    except Exception as e:
        print(f"Error calculating speedup for {t['name']}: {e}")

# Run parallel benchmarks with separated I/O and computation timing
print("\nRunning parallel benchmarks with separated timing...")
for t in test_cases:
    io_results = []
    compute_results = []
    total_results = []
    
    for threads in test_threads:
        print(f"Testing {t['name']} with {threads} threads (separated timing)...")
        
        # Benchmark with separated timing
        try:
            # We can't use timeit directly since we need to capture the separated times
            # So we'll run it manually a few times and average the results
            io_times = []
            compute_times = []
            total_times = []
            
            for _ in range(num_repeats):
                for _ in range(num_executions):
                    _, io_time, compute_time, total_time = polars_bio_base_content_parallel_separated(
                        t["file_path"], threads
                    )
                    io_times.append(io_time)
                    compute_times.append(compute_time)
                    total_times.append(total_time)
            
            # Calculate statistics
            io_result = {
                "name": f"I/O time ({threads} threads)",
                "min": min(io_times),
                "max": max(io_times),
                "mean": np.mean(io_times),
                "threads": threads
            }
            io_results.append(io_result)
            
            compute_result = {
                "name": f"Compute time ({threads} threads)",
                "min": min(compute_times),
                "max": max(compute_times),
                "mean": np.mean(compute_times),
                "threads": threads
            }
            compute_results.append(compute_result)
            
            total_result = {
                "name": f"Total time ({threads} threads)",
                "min": min(total_times),
                "max": max(total_times),
                "mean": np.mean(total_times),
                "threads": threads
            }
            total_results.append(total_result)
            
        except Exception as e:
            print(f"Error benchmarking with separated timing ({threads} threads): {e}")
        
        # Create Rich table for the separated timing results
        table = Table(title=f"Separated Timing Results - {t['name']} ({threads} threads)", box=MARKDOWN)
        table.add_column("Operation", justify="left", style="cyan", no_wrap=True)
        table.add_column("Min (s)", justify="right", style="green")
        table.add_column("Max (s)", justify="right", style="green")
        table.add_column("Mean (s)", justify="right", style="green")
        table.add_column("Threads", justify="right", style="magenta")
        
        # Add rows to the table
        if io_results:
            table.add_row(
                io_results[-1]["name"],
                f"{io_results[-1]['min']:.6f}",
                f"{io_results[-1]['max']:.6f}",
                f"{io_results[-1]['mean']:.6f}",
                str(io_results[-1]["threads"])
            )
        
        if compute_results:
            table.add_row(
                compute_results[-1]["name"],
                f"{compute_results[-1]['min']:.6f}",
                f"{compute_results[-1]['max']:.6f}",
                f"{compute_results[-1]['mean']:.6f}",
                str(compute_results[-1]["threads"])
            )
        
        if total_results:
            table.add_row(
                total_results[-1]["name"],
                f"{total_results[-1]['min']:.6f}",
                f"{total_results[-1]['max']:.6f}",
                f"{total_results[-1]['mean']:.6f}",
                str(total_results[-1]["threads"])
            )
        
        # Display the table
        print(table)
        
        # Save results to JSON
        separated_results = {
            "test_case": t["name"],
            "description": t["description"],
            "threads": threads,
            "io_results": [io_results[-1]] if io_results else [],
            "compute_results": [compute_results[-1]] if compute_results else [],
            "total_results": [total_results[-1]] if total_results else []
        }
        json.dump(separated_results, open(f"results/base_content_{t['name']}_{threads}_threads_separated.json", "w"))
    
    # After collecting all results for different thread counts, create a comparison table
    if io_results and compute_results and total_results:
        # Create Rich table for I/O speedup comparison
        io_table = Table(title=f"I/O Speedup Comparison - {t['name']}", box=MARKDOWN)
        io_table.add_column("Threads", justify="right", style="cyan")
        io_table.add_column("I/O Time (s)", justify="right", style="green")
        io_table.add_column("I/O Speedup", justify="right", style="magenta")
        
        # Calculate I/O speedups
        io_baseline = io_results[0]["mean"]  # 1 thread result
        for result in io_results:
            io_speedup = io_baseline / result["mean"] if result["mean"] > 0 else 0
            io_table.add_row(
                str(result["threads"]),
                f"{result['mean']:.6f}",
                f"{io_speedup:.2f}x"
            )
        
        # Display the I/O speedup table
        print(io_table)
        
        # Create Rich table for compute speedup comparison
        compute_table = Table(title=f"Compute Speedup Comparison - {t['name']}", box=MARKDOWN)
        compute_table.add_column("Threads", justify="right", style="cyan")
        compute_table.add_column("Compute Time (s)", justify="right", style="green")
        compute_table.add_column("Compute Speedup", justify="right", style="magenta")
        
        # Calculate compute speedups
        compute_baseline = compute_results[0]["mean"]  # 1 thread result
        for result in compute_results:
            compute_speedup = compute_baseline / result["mean"] if result["mean"] > 0 else 0
            compute_table.add_row(
                str(result["threads"]),
                f"{result['mean']:.6f}",
                f"{compute_speedup:.2f}x"
            )
        
        # Display the compute speedup table
        print(compute_table)
        
        # Create Rich table for total speedup comparison
        total_table = Table(title=f"Total Speedup Comparison - {t['name']}", box=MARKDOWN)
        total_table.add_column("Threads", justify="right", style="cyan")
        total_table.add_column("Total Time (s)", justify="right", style="green")
        total_table.add_column("Total Speedup", justify="right", style="magenta")
        
        # Calculate total speedups
        total_baseline = total_results[0]["mean"]  # 1 thread result
        for result in total_results:
            total_speedup = total_baseline / result["mean"] if result["mean"] > 0 else 0
            total_table.add_row(
                str(result["threads"]),
                f"{result['mean']:.6f}",
                f"{total_speedup:.2f}x"
            )
        
        # Display the total speedup table
        print(total_table)
        
        # Save all speedup results to JSON
        speedup_comparison = {
            "test_case": t["name"],
            "description": t["description"],
            "io_results": io_results,
            "compute_results": compute_results,
            "total_results": total_results
        }
        json.dump(speedup_comparison, open(f"results/base_content_{t['name']}_separated_speedup.json", "w"))
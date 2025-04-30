import os
import time
import json
import numpy as np
import polars as pl
import polars_bio as pb
from rich import print
from rich.box import MARKDOWN
from rich.table import Table
import subprocess
import tempfile
import multiprocessing
from pathlib import Path

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Set environment variables
os.environ["POLARS_MAX_THREADS"] = "1"  # For single-threaded tests
BENCH_DATA_ROOT = os.getenv("BENCH_DATA_ROOT", '/home/dbartosiak/praca/polars-bio/tests/data/qc/example.fastq')

if BENCH_DATA_ROOT is None:
    raise ValueError("BENCH_DATA_ROOT is not set")

# Test parameters
num_repeats = 3
num_executions = 3
test_threads = [1, 8]  # For parallel tests

# Test cases - adjust paths as needed
test_cases = [{
        "file_path": f"{BENCH_DATA_ROOT}",
        "name": "large",
        "description": "Large FASTQ file (~1GB)"
    }
]

# Function to run a single benchmark in a separate process
def run_single_benchmark(test_case, threads, repeats, executions):
    """Run a single benchmark with specified parameters and return results"""
    result_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    result_file.close()
    
    # Create a temporary script file
    script_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py')
    script_file.write(f"""
import time
import polars as pl
import polars_bio as pb
import json

# Run the benchmark with {threads} threads
file_path = "{test_case['file_path']}"
num_threads = {threads}
num_repeats = {repeats}
num_executions = {executions}

io_times = []
compute_times = []
total_times = []

for _ in range(num_repeats):
    for _ in range(num_executions):
        # Load data
        start_io = time.time()
        # Use read_fastq instead of read_csv for FASTQ files
        df = pb.read_fastq(file_path).collect()
        io_time = time.time() - start_io
        
        # Process data
        start_compute = time.time()
        # Extract sequence column for base content analysis
        sequences_df = df.select("sequence")
        result = pb.qc.base_content_parallel(sequences_df, num_threads=num_threads)
        compute_time = time.time() - start_compute
        
        total_time = io_time + compute_time
        
        io_times.append(io_time)
        compute_times.append(compute_time)
        total_times.append(total_time)

# Calculate statistics
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

# Save results
with open("{result_file.name}", "w") as f:
    json.dump(results, f)
""")
    script_file.close()
    
    # Run the script in a separate process
    try:
        subprocess.run(["python", script_file.name], check=True)
        
        # Read results
        with open(result_file.name, 'r') as f:
            results = json.load(f)
            
        # Clean up temporary files
        os.unlink(script_file.name)
        os.unlink(result_file.name)
        
        return results
    except Exception as e:
        print(f"Error running benchmark with {threads} threads: {e}")
        # Clean up temporary files even if there's an error
        if os.path.exists(script_file.name):
            os.unlink(script_file.name)
        if os.path.exists(result_file.name):
            os.unlink(result_file.name)
        return None

# Function to create a formatted table of results
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

# Main benchmark loop
for t in test_cases:
    print(f"Testing {t['name']}...")
    
    # Prepare result structure
    all_results = {
        "test_case": t["name"],
        "description": t["description"],
        "io_results": [],
        "compute_results": [],
        "total_results": []
    }
    
    # Run benchmarks for each thread count
    benchmark_results = []
    for threads in test_threads:
        print(f"Testing {t['name']} with {threads} threads in a separate process...")
        results = run_single_benchmark(t, threads, num_repeats, num_executions)
        if results:
            benchmark_results.append((threads, results))

    # Sort results by thread count
    benchmark_results.sort(key=lambda x: x[0])
    
    # Process results
    for threads, result in benchmark_results:
        all_results["io_results"].append({
            "name": f"I/O time ({threads} threads)",
            "min": result["io"]["min"],
            "max": result["io"]["max"],
            "mean": result["io"]["mean"],
            "threads": threads
        })
        
        all_results["compute_results"].append({
            "name": f"Compute time ({threads} threads)",
            "min": result["compute"]["min"],
            "max": result["compute"]["max"],
            "mean": result["compute"]["mean"],
            "threads": threads
        })
        
        all_results["total_results"].append({
            "name": f"Total time ({threads} threads)",
            "min": result["total"]["min"],
            "max": result["total"]["max"],
            "mean": result["total"]["mean"],
            "threads": threads
        })
    
    # Create and display tables
    if all_results["io_results"] and all_results["compute_results"] and all_results["total_results"]:
        # Create and display I/O speedup table
        io_table = create_speedup_table(
            all_results["io_results"], 
            0,  # baseline is first result (1 thread)
            f"I/O Speedup Comparison - {t['name']}", 
            "I/O"
        )
        print(io_table)
        
        # Create and display compute speedup table
        compute_table = create_speedup_table(
            all_results["compute_results"], 
            0,  # baseline is first result (1 thread)
            f"Compute Speedup Comparison - {t['name']}", 
            "Compute"
        )
        print(compute_table)
        
        # Create and display total speedup table
        total_table = create_speedup_table(
            all_results["total_results"], 
            0,  # baseline is first result (1 thread)
            f"Total Speedup Comparison - {t['name']}", 
            "Total"
        )
        print(total_table)
        
        # Save all speedup results to JSON
        result_file = Path("results") / f"base_content_{t['name']}_separated_speedup.json"
        with open(result_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Results saved to {result_file}")

use datafusion::arrow::array::{Array, ArrayRef, Float64Array, Int32Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use rayon::prelude::*; // Import Rayon's parallel iterator traits

/// Calculates base content percentages for each position in sequences
pub fn calculate_base_content(sequences: &StringArray) -> Result<RecordBatch> {
    let seq_count = sequences.len() as f64;
    if seq_count == 0.0 {
        return create_empty_result();
    }

    // Find the maximum sequence length
    let max_length = sequences
        .iter()
        .filter_map(|seq| seq.map(|s| s.len()))
        .max()
        .unwrap_or(0);

    if max_length == 0 {
        return create_empty_result();
    }

    // Initialize counters for each base at each position
    let mut a_counts = vec![0.0; max_length];
    let mut c_counts = vec![0.0; max_length];
    let mut g_counts = vec![0.0; max_length];
    let mut t_counts = vec![0.0; max_length];
    let mut n_counts = vec![0.0; max_length];

    // Count bases at each position
    for i in 0..sequences.len() {
        // Check if the value is not null
        if sequences.is_valid(i) {
            let seq = sequences.value(i);
            for (pos, base) in seq.chars().enumerate() {
                match base.to_ascii_uppercase() {
                    'A' => a_counts[pos] += 1.0,
                    'C' => c_counts[pos] += 1.0,
                    'G' => g_counts[pos] += 1.0,
                    'T' => t_counts[pos] += 1.0,
                    _ => n_counts[pos] += 1.0,
                }
            }
        }
    }

    // Convert counts to percentages
    let positions: Vec<i32> = (0..max_length as i32).collect();
    let a_percentages: Vec<f64> = a_counts.iter().map(|&count| (count / seq_count) * 100.0).collect();
    let c_percentages: Vec<f64> = c_counts.iter().map(|&count| (count / seq_count) * 100.0).collect();
    let g_percentages: Vec<f64> = g_counts.iter().map(|&count| (count / seq_count) * 100.0).collect();
    let t_percentages: Vec<f64> = t_counts.iter().map(|&count| (count / seq_count) * 100.0).collect();
    let n_percentages: Vec<f64> = n_counts.iter().map(|&count| (count / seq_count) * 100.0).collect();

    // Create Arrow arrays
    let position_array = Arc::new(Int32Array::from(positions)) as ArrayRef;
    let a_array = Arc::new(Float64Array::from(a_percentages)) as ArrayRef;
    let c_array = Arc::new(Float64Array::from(c_percentages)) as ArrayRef;
    let g_array = Arc::new(Float64Array::from(g_percentages)) as ArrayRef;
    let t_array = Arc::new(Float64Array::from(t_percentages)) as ArrayRef;
    let n_array = Arc::new(Float64Array::from(n_percentages)) as ArrayRef;

    // Create schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("position", DataType::Int32, false),
        Field::new("A", DataType::Float64, false),
        Field::new("C", DataType::Float64, false),
        Field::new("G", DataType::Float64, false),
        Field::new("T", DataType::Float64, false),
        Field::new("N", DataType::Float64, false),
    ]));

    // Create record batch
    let record_batch = RecordBatch::try_new(
        schema,
        vec![position_array, a_array, c_array, g_array, t_array, n_array],
    ).map_err(|e| DataFusionError::ArrowError(e, None))?;

    Ok(record_batch)
}

fn create_empty_result() -> Result<RecordBatch> {
    // Create an empty schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("position", DataType::Int32, false),
        Field::new("A", DataType::Float64, false),
        Field::new("C", DataType::Float64, false),
        Field::new("G", DataType::Float64, false),
        Field::new("T", DataType::Float64, false),
        Field::new("N", DataType::Float64, false),
    ]));

    // Create empty arrays
    let position_array = Arc::new(Int32Array::from(Vec::<i32>::new())) as ArrayRef;
    let a_array = Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef;
    let c_array = Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef;
    let g_array = Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef;
    let t_array = Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef;
    let n_array = Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef;

    // Create record batch
    let record_batch = RecordBatch::try_new(
        schema,
        vec![position_array, a_array, c_array, g_array, t_array, n_array],
    ).map_err(|e| DataFusionError::ArrowError(e, None))?;

    Ok(record_batch)
}

pub fn calculate_base_content_parallel(sequences: &StringArray, num_threads: usize) -> Result<RecordBatch> {
    let seq_count = sequences.len() as f64;
    if seq_count == 0.0 {
        return create_empty_result();
    }

    // Find the maximum sequence length
    let max_length = sequences
        .iter()
        .filter_map(|seq| seq.map(|s| s.len()))
        .max()
        .unwrap_or(0);

    if max_length == 0 {
        return create_empty_result();
    }

    // Use at least 1 thread, but no more than the number of sequences or available CPUs
    let available_cpus = num_cpus::get();
    let num_threads = std::cmp::min(
        std::cmp::min(std::cmp::max(num_threads, 1), sequences.len()),
        available_cpus
    );
    
    // For very small datasets, just use the single-threaded version
    if sequences.len() < 1000 || num_threads == 1 {
        return calculate_base_content(sequences);
    }
    
    // Calculate chunk size for each thread - ensure it's at least 100 sequences per thread
    let min_chunk_size = 100;
    let chunk_size = std::cmp::max(
        min_chunk_size,
        (sequences.len() + num_threads - 1) / num_threads
    );
    
    // Recalculate number of threads based on chunk size
    let actual_num_threads = (sequences.len() + chunk_size - 1) / chunk_size;
    
    // Create a thread pool for better reuse of threads
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(actual_num_threads)
        .build()
        .map_err(|e| DataFusionError::Execution(format!("Failed to build thread pool: {}", e)))?;
    
    // Create vectors to store results from each thread
    let mut all_results = Vec::with_capacity(actual_num_threads);
    
    // Process data in parallel using the thread pool
    pool.install(|| {
        all_results = (0..actual_num_threads)
            .into_par_iter() // Now this will work with the proper import
            .map(|thread_id| {
                let start = thread_id * chunk_size;
                let end = std::cmp::min(start + chunk_size, sequences.len());
                
                // Skip empty chunks
                if start >= end {
                    return (vec![0.0; max_length], vec![0.0; max_length], 
                            vec![0.0; max_length], vec![0.0; max_length], 
                            vec![0.0; max_length]);
                }
                
                // Create local counters for this thread
                let mut local_a_counts = vec![0.0; max_length];
                let mut local_c_counts = vec![0.0; max_length];
                let mut local_g_counts = vec![0.0; max_length];
                let mut local_t_counts = vec![0.0; max_length];
                let mut local_n_counts = vec![0.0; max_length];
                
                // Process assigned sequences
                for i in start..end {
                    if sequences.is_valid(i) {
                        let seq = sequences.value(i);
                        for (pos, base) in seq.chars().enumerate() {
                            match base.to_ascii_uppercase() {
                                'A' => local_a_counts[pos] += 1.0,
                                'C' => local_c_counts[pos] += 1.0,
                                'G' => local_g_counts[pos] += 1.0,
                                'T' => local_t_counts[pos] += 1.0,
                                _ => local_n_counts[pos] += 1.0,
                            }
                        }
                    }
                }
                
                (local_a_counts, local_c_counts, local_g_counts, local_t_counts, local_n_counts)
            })
            .collect();
    });
    
    // Initialize final count arrays
    let mut a_counts = vec![0.0; max_length];
    let mut c_counts = vec![0.0; max_length];
    let mut g_counts = vec![0.0; max_length];
    let mut t_counts = vec![0.0; max_length];
    let mut n_counts = vec![0.0; max_length];
    
    // Merge results from all threads
    for (thread_a_counts, thread_c_counts, thread_g_counts, thread_t_counts, thread_n_counts) in all_results {
        for i in 0..max_length {
            a_counts[i] += thread_a_counts[i];
            c_counts[i] += thread_c_counts[i];
            g_counts[i] += thread_g_counts[i];
            t_counts[i] += thread_t_counts[i];
            n_counts[i] += thread_n_counts[i];
        }
    }

    // Convert counts to percentages
    let positions: Vec<i32> = (0..max_length as i32).collect();
    let a_percentages: Vec<f64> = a_counts.iter().map(|&count| (count / seq_count) * 100.0).collect();
    let c_percentages: Vec<f64> = c_counts.iter().map(|&count| (count / seq_count) * 100.0).collect();
    let g_percentages: Vec<f64> = g_counts.iter().map(|&count| (count / seq_count) * 100.0).collect();
    let t_percentages: Vec<f64> = t_counts.iter().map(|&count| (count / seq_count) * 100.0).collect();
    let n_percentages: Vec<f64> = n_counts.iter().map(|&count| (count / seq_count) * 100.0).collect();

    // Create Arrow arrays
    let position_array = Arc::new(Int32Array::from(positions)) as ArrayRef;
    let a_array = Arc::new(Float64Array::from(a_percentages)) as ArrayRef;
    let c_array = Arc::new(Float64Array::from(c_percentages)) as ArrayRef;
    let g_array = Arc::new(Float64Array::from(g_percentages)) as ArrayRef;
    let t_array = Arc::new(Float64Array::from(t_percentages)) as ArrayRef;
    let n_array = Arc::new(Float64Array::from(n_percentages)) as ArrayRef;

    // Create schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("position", DataType::Int32, false),
        Field::new("A", DataType::Float64, false),
        Field::new("C", DataType::Float64, false),
        Field::new("G", DataType::Float64, false),
        Field::new("T", DataType::Float64, false),
        Field::new("N", DataType::Float64, false),
    ]));

    // Create record batch
    let record_batch = RecordBatch::try_new(
        schema,
        vec![position_array, a_array, c_array, g_array, t_array, n_array],
    ).map_err(|e| DataFusionError::ArrowError(e, None))?;

    Ok(record_batch)
}
use datafusion::arrow::array::{Array, ArrayRef, Float64Array, Int32Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use std::sync::Arc;
use rayon::prelude::*;
use std::sync::Once;
use std::cmp;
use std::arch::x86_64::*;

static THREAD_POOL_INIT: Once = Once::new();

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
            for (pos, base) in seq.bytes().enumerate() {
                match base {
                    b'A' | b'a' => a_counts[pos] += 1.0,
                    b'C' | b'c' => c_counts[pos] += 1.0,
                    b'G' | b'g' => g_counts[pos] += 1.0,
                    b'T' | b't' => t_counts[pos] += 1.0,
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

    create_result_batch(max_length, positions, a_percentages, c_percentages, g_percentages, t_percentages, n_percentages)
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

fn create_result_batch(
    max_length: usize,
    positions: Vec<i32>,
    a_percentages: Vec<f64>,
    c_percentages: Vec<f64>,
    g_percentages: Vec<f64>,
    t_percentages: Vec<f64>,
    n_percentages: Vec<f64>,
) -> Result<RecordBatch> {
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

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn count_bases_simd(seq: &[u8], a_counts: &mut [f64], c_counts: &mut [f64], g_counts: &mut [f64], t_counts: &mut [f64], n_counts: &mut [f64]) {
    // Sprawdzamy, czy CPU obsługuje AVX2
    if is_x86_feature_detected!("avx2") {
        // Przetwarzamy sekwencję po 16 bajtów na raz używając AVX2
        let mut pos = 0;
        let len = seq.len();
        
        // Maski do porównania (wartości ASCII dla A, C, G, T)
        let a_mask = _mm256_set1_epi8(b'A' as i8);
        let a_lower_mask = _mm256_set1_epi8(b'a' as i8);
        let c_mask = _mm256_set1_epi8(b'C' as i8);
        let c_lower_mask = _mm256_set1_epi8(b'c' as i8);
        let g_mask = _mm256_set1_epi8(b'G' as i8);
        let g_lower_mask = _mm256_set1_epi8(b'g' as i8);
        let t_mask = _mm256_set1_epi8(b'T' as i8);
        let t_lower_mask = _mm256_set1_epi8(b't' as i8);
        
        // Przetwarzamy po 32 bajty na raz
        while pos + 32 <= len {
            // Ładujemy 32 bajty z sekwencji
            let data = _mm256_loadu_si256(seq[pos..].as_ptr() as *const __m256i);
            
            // Porównujemy z maskami dla każdej zasady
            let a_match = _mm256_or_si256(
                _mm256_cmpeq_epi8(data, a_mask),
                _mm256_cmpeq_epi8(data, a_lower_mask)
            );
            let c_match = _mm256_or_si256(
                _mm256_cmpeq_epi8(data, c_mask),
                _mm256_cmpeq_epi8(data, c_lower_mask)
            );
            let g_match = _mm256_or_si256(
                _mm256_cmpeq_epi8(data, g_mask),
                _mm256_cmpeq_epi8(data, g_lower_mask)
            );
            let t_match = _mm256_or_si256(
                _mm256_cmpeq_epi8(data, t_mask),
                _mm256_cmpeq_epi8(data, t_lower_mask)
            );
            
            // Konwertujemy maski na bajty (0xFF dla dopasowania, 0x00 dla braku)
            let mut a_result = [0u8; 32];
            let mut c_result = [0u8; 32];
            let mut g_result = [0u8; 32];
            let mut t_result = [0u8; 32];
            
            _mm256_storeu_si256(a_result.as_mut_ptr() as *mut __m256i, a_match);
            _mm256_storeu_si256(c_result.as_mut_ptr() as *mut __m256i, c_match);
            _mm256_storeu_si256(g_result.as_mut_ptr() as *mut __m256i, g_match);
            _mm256_storeu_si256(t_result.as_mut_ptr() as *mut __m256i, t_match);
            
            // Aktualizujemy liczniki
            for i in 0..32 {
                if pos + i < len {
                    if a_result[i] != 0 {
                        a_counts[pos + i] += 1.0;
                    } else if c_result[i] != 0 {
                        c_counts[pos + i] += 1.0;
                    } else if g_result[i] != 0 {
                        g_counts[pos + i] += 1.0;
                    } else if t_result[i] != 0 {
                        t_counts[pos + i] += 1.0;
                    } else {
                        n_counts[pos + i] += 1.0;
                    }
                }
            }
            
            pos += 32;
        }
        
        // Przetwarzamy pozostałe bajty standardową metodą
        for i in pos..len {
            match seq[i] {
                b'A' | b'a' => a_counts[i] += 1.0,
                b'C' | b'c' => c_counts[i] += 1.0,
                b'G' | b'g' => g_counts[i] += 1.0,
                b'T' | b't' => t_counts[i] += 1.0,
                _ => n_counts[i] += 1.0,
            }
        }
    } else {
        // Fallback dla CPU bez AVX2
        for (i, &byte) in seq.iter().enumerate() {
            match byte {
                b'A' | b'a' => a_counts[i] += 1.0,
                b'C' | b'c' => c_counts[i] += 1.0,
                b'G' | b'g' => g_counts[i] += 1.0,
                b'T' | b't' => t_counts[i] += 1.0,
                _ => n_counts[i] += 1.0,
            }
        }
    }
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

    // For very small datasets, just use the single-threaded version
    if sequences.len() < 1000 || num_threads <= 1 {
        return calculate_base_content(sequences);
    }
    
    // Set the number of threads for Rayon only if not already initialized
    let mut thread_pool_error = None;
    THREAD_POOL_INIT.call_once(|| {
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global() 
        {
            thread_pool_error = Some(DataFusionError::Execution(
                format!("Failed to set global thread pool: {}", e)
            ));
        }
    });
    
    // Check if there was an error during initialization
    if let Some(err) = thread_pool_error {
        return Err(err);
    }
    
    // Calculate chunk size for better work distribution
    let chunk_size = cmp::max(
        100, // Minimum chunk size
        (sequences.len() + num_threads - 1) / num_threads
    );
    
    // Pre-allocate vectors for all sequences to avoid reallocation
    let mut all_sequences: Vec<(&str, usize)> = Vec::with_capacity(sequences.len());
    for i in 0..sequences.len() {
        if sequences.is_valid(i) {
            all_sequences.push((sequences.value(i), i));
        }
    }
    
    // Process data in parallel using Rayon
    let (a_counts, c_counts, g_counts, t_counts, n_counts) = all_sequences
        .par_chunks(chunk_size)
        .map(|chunk| {
            // Create local counters for this chunk
            let mut local_a_counts = vec![0.0; max_length];
            let mut local_c_counts = vec![0.0; max_length];
            let mut local_g_counts = vec![0.0; max_length];
            let mut local_t_counts = vec![0.0; max_length];
            let mut local_n_counts = vec![0.0; max_length];
            
            // Process sequences in this chunk
            for &(seq, _) in chunk {
                // Use SIMD acceleration if available
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    count_bases_simd(
                        seq.as_bytes(),
                        &mut local_a_counts,
                        &mut local_c_counts,
                        &mut local_g_counts,
                        &mut local_t_counts,
                        &mut local_n_counts
                    );
                }
                
                #[cfg(not(target_arch = "x86_64"))]
                {
                    // Fallback for non-x86_64 architectures
                    for (pos, &byte) in seq.as_bytes().iter().enumerate() {
                        match byte {
                            b'A' | b'a' => local_a_counts[pos] += 1.0,
                            b'C' | b'c' => local_c_counts[pos] += 1.0,
                            b'G' | b'g' => local_g_counts[pos] += 1.0,
                            b'T' | b't' => local_t_counts[pos] += 1.0,
                            _ => local_n_counts[pos] += 1.0,
                        }
                    }
                }
            }
            
            (local_a_counts, local_c_counts, local_g_counts, local_t_counts, local_n_counts)
        })
        .reduce(
            || (vec![0.0; max_length], vec![0.0; max_length], vec![0.0; max_length], vec![0.0; max_length], vec![0.0; max_length]),
            |mut acc, chunk_counts| {
                let (chunk_a, chunk_c, chunk_g, chunk_t, chunk_n) = chunk_counts;
                
                // Vectorized addition of arrays
                for i in 0..max_length {
                    acc.0[i] += chunk_a[i];
                    acc.1[i] += chunk_c[i];
                    acc.2[i] += chunk_g[i];
                    acc.3[i] += chunk_t[i];
                    acc.4[i] += chunk_n[i];
                }
                acc
            }
        );

    // Convert counts to percentages
    let positions: Vec<i32> = (0..max_length as i32).collect();
    
    // Use parallel iterators for percentage calculation
    let a_percentages: Vec<f64> = a_counts.par_iter().map(|&count| (count / seq_count) * 100.0).collect();
    let c_percentages: Vec<f64> = c_counts.par_iter().map(|&count| (count / seq_count) * 100.0).collect();
    let g_percentages: Vec<f64> = g_counts.par_iter().map(|&count| (count / seq_count) * 100.0).collect();
    let t_percentages: Vec<f64> = t_counts.par_iter().map(|&count| (count / seq_count) * 100.0).collect();
    let n_percentages: Vec<f64> = n_counts.par_iter().map(|&count| (count / seq_count) * 100.0).collect();

    create_result_batch(max_length, positions, a_percentages, c_percentages, g_percentages, t_percentages, n_percentages)
}
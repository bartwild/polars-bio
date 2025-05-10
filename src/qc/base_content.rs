use arrow_array::{Array, ArrayRef, Float64Array, Int32Array, StringArray};
use arrow_schema::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use std::arch::x86_64::*;
use std::sync::Arc;


static THREAD_POOL_INIT: OnceCell<Result<(), DataFusionError>> = OnceCell::new();

fn create_empty_result() -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("position", DataType::Int32, false),
        Field::new("A", DataType::Float64, false),
        Field::new("C", DataType::Float64, false),
        Field::new("G", DataType::Float64, false),
        Field::new("T", DataType::Float64, false),
        Field::new("N", DataType::Float64, false),
    ]));

    let empty = || Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef;

    let record_batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(Vec::<i32>::new())) as ArrayRef,
            empty(), empty(), empty(), empty(), empty(),
        ],
    )?;

    Ok(record_batch)
}

fn create_result_batch(
    positions: Vec<i32>,
    a: Vec<f64>,
    c: Vec<f64>,
    g: Vec<f64>,
    t: Vec<f64>,
    n: Vec<f64>,
) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("position", DataType::Int32, false),
        Field::new("A", DataType::Float64, false),
        Field::new("C", DataType::Float64, false),
        Field::new("G", DataType::Float64, false),
        Field::new("T", DataType::Float64, false),
        Field::new("N", DataType::Float64, false),
    ]));

    let record_batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(positions)) as ArrayRef,
            Arc::new(Float64Array::from(a)) as ArrayRef,
            Arc::new(Float64Array::from(c)) as ArrayRef,
            Arc::new(Float64Array::from(g)) as ArrayRef,
            Arc::new(Float64Array::from(t)) as ArrayRef,
            Arc::new(Float64Array::from(n)) as ArrayRef,
        ],
    )?;

    Ok(record_batch)
}

#[target_feature(enable = "avx2")]
unsafe fn count_bases_simd_slice(
    seq: &[u8],
    a: &mut [f64],
    c: &mut [f64],
    g: &mut [f64],
    t: &mut [f64],
    n: &mut [f64],
) {
    let len = seq.len();
    let mut i = 0;

    let a_mask = _mm256_set1_epi8(b'A' as i8);
    let a_lower_mask = _mm256_set1_epi8(b'a' as i8);
    let c_mask = _mm256_set1_epi8(b'C' as i8);
    let c_lower_mask = _mm256_set1_epi8(b'c' as i8);
    let g_mask = _mm256_set1_epi8(b'G' as i8);
    let g_lower_mask = _mm256_set1_epi8(b'g' as i8);
    let t_mask = _mm256_set1_epi8(b'T' as i8);
    let t_lower_mask = _mm256_set1_epi8(b't' as i8);

    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(seq[i..].as_ptr() as *const __m256i);

        let a_cmp = _mm256_or_si256(_mm256_cmpeq_epi8(chunk, a_mask), _mm256_cmpeq_epi8(chunk, a_lower_mask));
        let c_cmp = _mm256_or_si256(_mm256_cmpeq_epi8(chunk, c_mask), _mm256_cmpeq_epi8(chunk, c_lower_mask));
        let g_cmp = _mm256_or_si256(_mm256_cmpeq_epi8(chunk, g_mask), _mm256_cmpeq_epi8(chunk, g_lower_mask));
        let t_cmp = _mm256_or_si256(_mm256_cmpeq_epi8(chunk, t_mask), _mm256_cmpeq_epi8(chunk, t_lower_mask));

        let mut a_bytes = [0u8; 32];
        let mut c_bytes = [0u8; 32];
        let mut g_bytes = [0u8; 32];
        let mut t_bytes = [0u8; 32];

        _mm256_storeu_si256(a_bytes.as_mut_ptr() as *mut __m256i, a_cmp);
        _mm256_storeu_si256(c_bytes.as_mut_ptr() as *mut __m256i, c_cmp);
        _mm256_storeu_si256(g_bytes.as_mut_ptr() as *mut __m256i, g_cmp);
        _mm256_storeu_si256(t_bytes.as_mut_ptr() as *mut __m256i, t_cmp);

        for j in 0..32 {
            let idx = i + j;
            if a_bytes[j] != 0 {
                a[idx] += 1.0;
            } else if c_bytes[j] != 0 {
                c[idx] += 1.0;
            } else if g_bytes[j] != 0 {
                g[idx] += 1.0;
            } else if t_bytes[j] != 0 {
                t[idx] += 1.0;
            } else {
                n[idx] += 1.0;
            }
        }

        i += 32;
    }

    for j in i..len {
        match seq[j] {
            b'A' | b'a' => a[j] += 1.0,
            b'C' | b'c' => c[j] += 1.0,
            b'G' | b'g' => g[j] += 1.0,
            b'T' | b't' => t[j] += 1.0,
            _ => n[j] += 1.0,
        }
    }
}


pub fn calculate_base_content(sequences: &StringArray, num_threads: usize) -> Result<RecordBatch> {
    if sequences.is_empty() {
        return create_empty_result();
    }

    let max_length = sequences.iter().filter_map(|s| s.map(|s| s.len())).max().unwrap_or(0);
    if max_length == 0 {
        return create_empty_result();
    }

    let result = THREAD_POOL_INIT.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .map_err(|e| DataFusionError::Execution(format!("Thread pool init failed: {}", e)))
    });
    
    if let Err(err) = result {
        return Err(DataFusionError::Execution(format!("{}", err)));
    }

    let all_sequences: Vec<&str> = (0..sequences.len())
        .filter(|&i| sequences.is_valid(i))
        .map(|i| sequences.value(i))
        .collect();

    let chunk_size = std::cmp::max(1, all_sequences.len() / num_threads);

    let (a, c, g, t, n, pos) = all_sequences
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut a = vec![0.0; max_length];
            let mut c = vec![0.0; max_length];
            let mut g = vec![0.0; max_length];
            let mut t = vec![0.0; max_length];
            let mut n = vec![0.0; max_length];
            let mut pos = vec![0.0; max_length];

            for &seq in chunk {
                let bytes = seq.as_bytes();
                let len = bytes.len();

                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        count_bases_simd_slice(bytes, &mut a, &mut c, &mut g, &mut t, &mut n);
                    }
                } else {
                    for i in 0..len {
                        match bytes[i] {
                            b'A' | b'a' => a[i] += 1.0,
                            b'C' | b'c' => c[i] += 1.0,
                            b'G' | b'g' => g[i] += 1.0,
                            b'T' | b't' => t[i] += 1.0,
                            _ => n[i] += 1.0,
                        }
                    }
                }

                for i in 0..len {
                    pos[i] += 1.0;
                }
            }

            (a, c, g, t, n, pos)
        })
        .reduce_with(|(mut a1, mut c1, mut g1, mut t1, mut n1, mut p1), (a2, c2, g2, t2, n2, p2)| {
            for i in 0..max_length {
                a1[i] += a2[i];
                c1[i] += c2[i];
                g1[i] += g2[i];
                t1[i] += t2[i];
                n1[i] += n2[i];
                p1[i] += p2[i];
            }
            (a1, c1, g1, t1, n1, p1)
        })
        .unwrap_or_else(|| {
            (
                vec![0.0; max_length],
                vec![0.0; max_length],
                vec![0.0; max_length],
                vec![0.0; max_length],
                vec![0.0; max_length],
                vec![0.0; max_length],
            )
        });

    let normalize = |counts: &[f64], pos: &[f64]| {
        counts
            .par_iter()
            .zip(pos.par_iter())
            .map(|(&count, &p)| if p > 0.0 { (count / p) * 100.0 } else { 0.0 })
            .collect()
    };

    let positions: Vec<i32> = (0..max_length as i32).collect();

    create_result_batch(
        positions,
        normalize(&a, &pos),
        normalize(&c, &pos),
        normalize(&g, &pos),
        normalize(&t, &pos),
        normalize(&n, &pos),
    )
}

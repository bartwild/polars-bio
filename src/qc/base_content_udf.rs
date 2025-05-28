use datafusion::arrow::array::{StringArray, Float64Array, UInt64Array, StructArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, Fields};
use datafusion::error::Result;
use datafusion::logical_expr::{Volatility, ColumnarValue, ScalarUDF, create_udf};
use arrow_array::Array;
use std::sync::Arc;
use arrow_array::RecordBatch;

pub fn create_base_content_udfs() -> Vec<ScalarUDF> {
    vec![
        create_base_content_udf(),
        create_base_content_by_position_udf(),
    ]
}

fn create_base_content_udf() -> ScalarUDF {
    let func = move |args: &[ColumnarValue]| -> Result<ColumnarValue> {
        let sequence_array = match &args[0] {
            ColumnarValue::Array(array) => array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Expected string array"),
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "Expected array argument".to_string(),
            )),
        };

        let num_rows = sequence_array.len();

        let mut max_length = 0;
        for i in 0..num_rows {
            if sequence_array.is_null(i) {
                continue;
            }

            let sequence = sequence_array.value(i);
            max_length = max_length.max(sequence.len());
        }

        let mut a_counts = vec![0u64; max_length];
        let mut c_counts = vec![0u64; max_length];
        let mut g_counts = vec![0u64; max_length];
        let mut t_counts = vec![0u64; max_length];
        let mut n_counts = vec![0u64; max_length];
        let mut total_counts = vec![0u64; max_length];

        for i in 0..num_rows {
            if sequence_array.is_null(i) {
                continue;
            }

            let sequence = sequence_array.value(i);
            for (pos, base) in sequence.chars().enumerate() {
                match base.to_ascii_uppercase() {
                    'A' => { a_counts[pos] += 1; total_counts[pos] += 1; },
                    'C' => { c_counts[pos] += 1; total_counts[pos] += 1; },
                    'G' => { g_counts[pos] += 1; total_counts[pos] += 1; },
                    'T' => { t_counts[pos] += 1; total_counts[pos] += 1; },
                    _ => { n_counts[pos] += 1; total_counts[pos] += 1; },
                }
            }
        }

        let mut position_builder = UInt64Array::builder(max_length);
        for i in 0..max_length {
            position_builder.append_value(i as u64);
        }
        let position_array = Arc::new(position_builder.finish());

        let mut a_builder = Float64Array::builder(max_length);
        let mut c_builder = Float64Array::builder(max_length);
        let mut g_builder = Float64Array::builder(max_length);
        let mut t_builder = Float64Array::builder(max_length);
        let mut n_builder = Float64Array::builder(max_length);

        for i in 0..max_length {
            let total = total_counts[i] as f64;
            if total > 0.0 {
                a_builder.append_value((a_counts[i] as f64) / total * 100.0);
                c_builder.append_value((c_counts[i] as f64) / total * 100.0);
                g_builder.append_value((g_counts[i] as f64) / total * 100.0);
                t_builder.append_value((t_counts[i] as f64) / total * 100.0);
                n_builder.append_value((n_counts[i] as f64) / total * 100.0);
            } else {
                a_builder.append_value(0.0);
                c_builder.append_value(0.0);
                g_builder.append_value(0.0);
                t_builder.append_value(0.0);
                n_builder.append_value(0.0);
            }
        }

        let a_array = Arc::new(a_builder.finish());
        let c_array = Arc::new(c_builder.finish());
        let g_array = Arc::new(g_builder.finish());
        let t_array = Arc::new(t_builder.finish());
        let n_array = Arc::new(n_builder.finish());

        let schema = Schema::new(vec![
            Field::new("position", DataType::UInt64, false),
            Field::new("A", DataType::Float64, false),
            Field::new("C", DataType::Float64, false),
            Field::new("G", DataType::Float64, false),
            Field::new("T", DataType::Float64, false),
            Field::new("N", DataType::Float64, false),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                position_array,
                a_array,
                c_array,
                g_array,
                t_array,
                n_array,
            ],
        )?;

        let fields = Fields::from(vec![
            Field::new("position", DataType::UInt64, false),
            Field::new("A", DataType::Float64, false),
            Field::new("C", DataType::Float64, false),
            Field::new("G", DataType::Float64, false),
            Field::new("T", DataType::Float64, false),
            Field::new("N", DataType::Float64, false),
        ]);

        let struct_array = StructArray::new(
            fields.clone(),
            batch.columns().to_vec(),
            None,
        );

        Ok(ColumnarValue::Array(Arc::new(struct_array)))
    };

    create_udf(
        "base_content",
        vec![DataType::Utf8],
        DataType::Struct(Fields::from(vec![
            Field::new("position", DataType::UInt64, false),
            Field::new("A", DataType::Float64, false),
            Field::new("C", DataType::Float64, false),
            Field::new("G", DataType::Float64, false),
            Field::new("T", DataType::Float64, false),
            Field::new("N", DataType::Float64, false),
        ])),
        Volatility::Immutable,
        Arc::new(func),
    )
}

fn create_base_content_by_position_udf() -> ScalarUDF {
    let func = move |args: &[ColumnarValue]| -> Result<ColumnarValue> {
        let sequence_array = match &args[0] {
            ColumnarValue::Array(array) => array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Expected string array"),
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "Expected array argument".to_string(),
            )),
        };

        let base_array = match &args[1] {
            ColumnarValue::Array(array) => array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Expected string array for base"),
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "Expected array argument for base".to_string(),
            )),
        };

        if base_array.len() == 0 {
            return Err(datafusion::error::DataFusionError::Execution(
                "Base parameter cannot be empty".to_string(),
            ));
        }

        let base_to_check = base_array.value(0).chars().next().unwrap_or('A').to_ascii_uppercase();

        let num_rows = sequence_array.len();

        let mut max_length = 0;
        for i in 0..num_rows {
            if sequence_array.is_null(i) {
                continue;
            }

            let sequence = sequence_array.value(i);
            max_length = max_length.max(sequence.len());
        }

        let mut base_counts = vec![0u64; max_length];
        let mut total_counts = vec![0u64; max_length];

        for i in 0..num_rows {
            if sequence_array.is_null(i) {
                continue;
            }

            let sequence = sequence_array.value(i);
            for (pos, base) in sequence.chars().enumerate() {
                if base.to_ascii_uppercase() == base_to_check {
                    base_counts[pos] += 1;
                }
                total_counts[pos] += 1;
            }
        }

        let mut position_builder = UInt64Array::builder(max_length);
        let mut percentage_builder = Float64Array::builder(max_length);

        for i in 0..max_length {
            position_builder.append_value(i as u64);

            let total = total_counts[i] as f64;
            if total > 0.0 {
                percentage_builder.append_value((base_counts[i] as f64) / total * 100.0);
            } else {
                percentage_builder.append_value(0.0);
            }
        }

        let percentage_array = Arc::new(percentage_builder.finish());

        Ok(ColumnarValue::Array(percentage_array))
    };

    create_udf(
        "base_content_by_position",
        vec![DataType::Utf8, DataType::Utf8],
        DataType::Float64,
        Volatility::Immutable,
        Arc::new(func),
    )
}
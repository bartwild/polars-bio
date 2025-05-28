use datafusion::arrow::array::{StringArray, Float64Array, UInt64Array, StructArray};
use datafusion::arrow::datatypes::{DataType, Field, Fields};
use datafusion::error::Result;
use datafusion::logical_expr::{Volatility, AggregateUDF, create_udaf};
use datafusion::physical_plan::Accumulator;
use datafusion::logical_expr::function::AccumulatorArgs;
use datafusion::scalar::ScalarValue;
use arrow::array::AsArray;
use arrow_array::Array;
use std::sync::Arc;

pub fn create_base_content_udaf() -> AggregateUDF {
    let accumulator_creator = move |_args: AccumulatorArgs<'_>| -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(BaseContentAccumulator::new()))
    };

    let return_type = Arc::new(DataType::Struct(Fields::from(vec![
        Field::new("position", DataType::UInt64, false),
        Field::new("A", DataType::Float64, false),
        Field::new("C", DataType::Float64, false),
        Field::new("G", DataType::Float64, false),
        Field::new("T", DataType::Float64, false),
        Field::new("N", DataType::Float64, false),
    ])));

    let state_type = vec![
        DataType::UInt64,
        DataType::List(Arc::new(Field::new("A", DataType::UInt64, false))),
        DataType::List(Arc::new(Field::new("C", DataType::UInt64, false))),
        DataType::List(Arc::new(Field::new("G", DataType::UInt64, false))),
        DataType::List(Arc::new(Field::new("T", DataType::UInt64, false))),
        DataType::List(Arc::new(Field::new("N", DataType::UInt64, false))),
        DataType::List(Arc::new(Field::new("Tot", DataType::UInt64, false))),
    ];

    create_udaf(
        "base_content",
        vec![DataType::Utf8],
        return_type,
        Volatility::Immutable,
        Arc::new(accumulator_creator),
        Arc::new(state_type),
    )
}

#[derive(Debug)]
struct BaseContentAccumulator {
    a_counts: Vec<u64>,
    c_counts: Vec<u64>,
    g_counts: Vec<u64>,
    t_counts: Vec<u64>,
    n_counts: Vec<u64>,
    total_counts: Vec<u64>,
    max_length: usize,
}

impl BaseContentAccumulator {
    fn new() -> Self {
        BaseContentAccumulator {
            a_counts: Vec::new(),
            c_counts: Vec::new(),
            g_counts: Vec::new(),
            t_counts: Vec::new(),
            n_counts: Vec::new(),
            total_counts: Vec::new(),
            max_length: 0,
        }
    }
}

impl Accumulator for BaseContentAccumulator {
    fn update_batch(&mut self, values: &[Arc<dyn Array>]) -> Result<()> {
        let sequence_array = values[0]
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Expected string array");

        let num_rows = sequence_array.len();

        for i in 0..num_rows {
            if sequence_array.is_null(i) {
                continue;
            }

            let sequence = sequence_array.value(i);
            self.max_length = self.max_length.max(sequence.len());
        }

        if self.a_counts.len() < self.max_length {
            self.a_counts.resize(self.max_length, 0);
            self.c_counts.resize(self.max_length, 0);
            self.g_counts.resize(self.max_length, 0);
            self.t_counts.resize(self.max_length, 0);
            self.n_counts.resize(self.max_length, 0);
            self.total_counts.resize(self.max_length, 0);
        }

        for i in 0..num_rows {
            if sequence_array.is_null(i) {
                continue;
            }

            let sequence = sequence_array.value(i);
            for (pos, base) in sequence.chars().enumerate() {
                match base.to_ascii_uppercase() {
                    'A' => { self.a_counts[pos] += 1; self.total_counts[pos] += 1; },
                    'C' => { self.c_counts[pos] += 1; self.total_counts[pos] += 1; },
                    'G' => { self.g_counts[pos] += 1; self.total_counts[pos] += 1; },
                    'T' => { self.t_counts[pos] += 1; self.total_counts[pos] += 1; },
                    _ => { self.n_counts[pos] += 1; self.total_counts[pos] += 1; },
                }
            }
        }

        Ok(())
    }

    fn merge_batch(&mut self, states: &[Arc<dyn Array>]) -> Result<()> {
        use arrow_array::{ListArray, UInt64Array};
        use arrow_array::types::UInt64Type;

        if states.len() != 7 {
            return Err(datafusion::error::DataFusionError::Internal(
                format!("Expected 7 state arrays, got {}", states.len())
            ));
        }

        let max_length_array = states[0]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("Expected UInt64Array for max_length");

        let a_counts_list = states[1]
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("Expected ListArray for a_counts");

        let c_counts_list = states[2]
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("Expected ListArray for c_counts");

        let g_counts_list = states[3]
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("Expected ListArray for g_counts");

        let t_counts_list = states[4]
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("Expected ListArray for t_counts");

        let n_counts_list = states[5]
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("Expected ListArray for n_counts");

        let total_counts_list = states[6]
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("Expected ListArray for total_counts");

        for row in 0..max_length_array.len() {
            let other_max_length = max_length_array.value(row) as usize;

            self.max_length = self.max_length.max(other_max_length);

            if self.a_counts.len() < self.max_length {
                self.a_counts.resize(self.max_length, 0);
                self.c_counts.resize(self.max_length, 0);
                self.g_counts.resize(self.max_length, 0);
                self.t_counts.resize(self.max_length, 0);
                self.n_counts.resize(self.max_length, 0);
                self.total_counts.resize(self.max_length, 0);
            }

            let a_counts = a_counts_list.value(row);
            let c_counts = c_counts_list.value(row);
            let g_counts = g_counts_list.value(row);
            let t_counts = t_counts_list.value(row);
            let n_counts = n_counts_list.value(row);
            let total_counts = total_counts_list.value(row);

            let a_counts = a_counts.as_primitive::<UInt64Type>();
            let c_counts = c_counts.as_primitive::<UInt64Type>();
            let g_counts = g_counts.as_primitive::<UInt64Type>();
            let t_counts = t_counts.as_primitive::<UInt64Type>();
            let n_counts = n_counts.as_primitive::<UInt64Type>();
            let total_counts = total_counts.as_primitive::<UInt64Type>();

            for i in 0..other_max_length {
                self.a_counts[i] += a_counts.value(i);
                self.c_counts[i] += c_counts.value(i);
                self.g_counts[i] += g_counts.value(i);
                self.t_counts[i] += t_counts.value(i);
                self.n_counts[i] += n_counts.value(i);
                self.total_counts[i] += total_counts.value(i);
            }
        }

        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        use arrow_array::UInt64Array;
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::{DataType, Field};
        use std::sync::Arc;

        let a_counts_array = UInt64Array::from(self.a_counts.clone());
        let c_counts_array = UInt64Array::from(self.c_counts.clone());
        let g_counts_array = UInt64Array::from(self.g_counts.clone());
        let t_counts_array = UInt64Array::from(self.t_counts.clone());
        let n_counts_array = UInt64Array::from(self.n_counts.clone());
        let total_counts_array = UInt64Array::from(self.total_counts.clone());

        let offsets = OffsetBuffer::new(vec![0, self.a_counts.len() as i32].into());


        let a_list_array = arrow_array::ListArray::try_new(
            Arc::new(Field::new("A", DataType::UInt64, false)),
            offsets.clone(),
            Arc::new(a_counts_array),
            None,
        ).unwrap();

        let c_list_array = arrow_array::ListArray::try_new(
            Arc::new(Field::new("C", DataType::UInt64, false)),
            offsets.clone(),
            Arc::new(c_counts_array),
            None,
        ).unwrap();

        let g_list_array = arrow_array::ListArray::try_new(
            Arc::new(Field::new("G", DataType::UInt64, false)),
            offsets.clone(),
            Arc::new(g_counts_array),
            None,
        ).unwrap();

        let t_list_array = arrow_array::ListArray::try_new(
            Arc::new(Field::new("T", DataType::UInt64, false)),
            offsets.clone(),
            Arc::new(t_counts_array),
            None,
        ).unwrap();

        let n_list_array = arrow_array::ListArray::try_new(
            Arc::new(Field::new("N", DataType::UInt64, false)),
            offsets.clone(),
            Arc::new(n_counts_array),
            None,
        ).unwrap();

        let total_list_array = arrow_array::ListArray::try_new(
            Arc::new(Field::new("Tot", DataType::UInt64, false)),
            offsets,
            Arc::new(total_counts_array),
            None,
        ).unwrap();

        let max_length_scalar = ScalarValue::UInt64(Some(self.max_length as u64));

        let mut state = Vec::with_capacity(7);
        state.push(max_length_scalar);
        state.push(ScalarValue::List(Arc::new(a_list_array)));
        state.push(ScalarValue::List(Arc::new(c_list_array)));
        state.push(ScalarValue::List(Arc::new(g_list_array)));
        state.push(ScalarValue::List(Arc::new(t_list_array)));
        state.push(ScalarValue::List(Arc::new(n_list_array)));
        state.push(ScalarValue::List(Arc::new(total_list_array)));

        Ok(state)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut position_builder = UInt64Array::builder(self.max_length);
        let mut a_builder = Float64Array::builder(self.max_length);
        let mut c_builder = Float64Array::builder(self.max_length);
        let mut g_builder = Float64Array::builder(self.max_length);
        let mut t_builder = Float64Array::builder(self.max_length);
        let mut n_builder = Float64Array::builder(self.max_length);

        for i in 0..self.max_length {
            position_builder.append_value(i as u64);

            let total = self.total_counts[i] as f64;
            if total > 0.0 {
                a_builder.append_value((self.a_counts[i] as f64) / total * 100.0);
                c_builder.append_value((self.c_counts[i] as f64) / total * 100.0);
                g_builder.append_value((self.g_counts[i] as f64) / total * 100.0);
                t_builder.append_value((self.t_counts[i] as f64) / total * 100.0);
                n_builder.append_value((self.n_counts[i] as f64) / total * 100.0);
            } else {
                a_builder.append_value(0.0);
                c_builder.append_value(0.0);
                g_builder.append_value(0.0);
                t_builder.append_value(0.0);
                n_builder.append_value(0.0);
            }
        }

        let position_array = Arc::new(position_builder.finish());
        let a_array = Arc::new(a_builder.finish());
        let c_array = Arc::new(c_builder.finish());
        let g_array = Arc::new(g_builder.finish());
        let t_array = Arc::new(t_builder.finish());
        let n_array = Arc::new(n_builder.finish());

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
            vec![
                position_array,
                a_array,
                c_array,
                g_array,
                t_array,
                n_array,
            ],
            None,
        );

        Ok(ScalarValue::Struct(Arc::new(struct_array)))
    }

    fn size(&self) -> usize {
        let vec_size = self.max_length * 6 * std::mem::size_of::<u64>();
        vec_size + std::mem::size_of::<Self>()
    }
}
use std::any::Any;
use std::sync::Arc;

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use async_stream::stream;
use futures::StreamExt;

use datafusion::common::{Result, Statistics};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{TableType, UserDefinedTableFunction};
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;

use crate::qc::base_content::calculate_base_content;

#[derive(Debug, Clone)]
pub struct BaseContentUDTF {
    schema: Arc<Schema>,
}

impl BaseContentUDTF {
    pub fn new() -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("position", DataType::Int32, false),
            Field::new("A", DataType::Float64, false),
            Field::new("C", DataType::Float64, false),
            Field::new("G", DataType::Float64, false),
            Field::new("T", DataType::Float64, false),
            Field::new("N", DataType::Float64, false),
        ]));
        BaseContentUDTF { schema }
    }
}

impl UserDefinedTableFunction for BaseContentUDTF {
    fn name(&self) -> &str {
        "base_content"
    }

    fn return_type(&self, _input_schema: &[Schema]) -> Result<Schema> {
        Ok(self.schema.as_ref().clone())
    }

    fn execute(
        &self,
        partitions: &[Vec<RecordBatch>],
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let schema = self.schema.clone();
        let mut result_batches = Vec::new();
        println!("This is a message from Rust!");
        for partition in partitions {
            for batch in partition {
                let sequence_col_idx = batch
                    .schema()
                    .index_of("sequence")
                    .map_err(|_| {
                        datafusion::error::DataFusionError::Execution(
                            "No 'sequence' column found in input".to_string(),
                        )
                    })?;

                let sequence_array = batch
                    .column(sequence_col_idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        datafusion::error::DataFusionError::Execution(
                            "'sequence' column is not a StringArray".to_string(),
                        )
                    })?;

                let num_threads = _context
                    .session_config()
                    .target_partitions()
                    .unwrap_or(1);
                let result_batch = calculate_base_content(sequence_array, num_threads)?;
                result_batches.push(result_batch);
            }
        }

        let stream = stream! {
            for batch in result_batches {
                yield Ok(batch);
            }
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, Box::pin(stream))))
    }

    fn supports_returning_stream(&self) -> bool {
        true
    }

    fn table_type(&self) -> TableType {
        TableType::Function
    }
}

#[derive(Debug, Clone)]
pub struct BaseContentExec {
    input: Arc<dyn ExecutionPlan>,
    schema: Arc<Schema>,
}

impl BaseContentExec {
    pub fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("position", DataType::Int32, false),
            Field::new("A", DataType::Float64, false),
            Field::new("C", DataType::Float64, false),
            Field::new("G", DataType::Float64, false),
            Field::new("T", DataType::Float64, false),
            Field::new("N", DataType::Float64, false),
        ]));

        Self { input, schema }
    }
}

impl ExecutionPlan for BaseContentExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn output_partitioning(&self) -> datafusion::physical_plan::Partitioning {
        self.input.output_partitioning()
    }

    fn output_ordering(&self) -> Option<&[datafusion::physical_expr::PhysicalSortExpr]> {
        None
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(children[0].clone())))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let schema = self.schema.clone();
        let input_stream = self.input.execute(partition, context)?;

        let stream = stream! {
            futures::pin_mut!(input_stream);
            while let Some(batch_result) = input_stream.next().await {
                let batch = batch_result?;

                let sequence_col_idx = batch
                    .schema()
                    .index_of("sequence")
                    .map_err(|_| {
                        datafusion::error::DataFusionError::Execution(
                            "No 'sequence' column found in input".to_string(),
                        )
                    })?;

                let sequence_array = batch
                    .column(sequence_col_idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        datafusion::error::DataFusionError::Execution(
                            "'sequence' column is not a StringArray".to_string(),
                        )
                    })?;
                let num_threads = _context
                    .session_config()
                    .target_partitions()
                    .unwrap_or(1);
                let result_batch = calculate_base_content(sequence_array, num_threads)?;
                yield Ok(result_batch);
            }
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, Box::pin(stream))))
    }

    fn statistics(&self) -> Statistics {
        Statistics::default()
    }
}

impl DisplayAs for BaseContentExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "BaseContentExec")
    }
}

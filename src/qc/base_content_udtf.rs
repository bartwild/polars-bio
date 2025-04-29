use std::any::Any;
use std::sync::Arc;
use arrow::array::{ArrayRef, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use async_stream::stream;
use datafusion::common::{Result, Statistics};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{TableType, UserDefinedTableFunction};
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, SendableRecordBatchStream as PhysicalSendableRecordBatchStream};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::Stream;
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
        
        // Process each partition
        let mut result_batches = Vec::new();
        
        for partition in partitions {
            for batch in partition {
                // Find the sequence column
                let sequence_col_idx = batch
                    .schema()
                    .fields()
                    .iter()
                    .position(|f| f.name() == "sequence")
                    .ok_or_else(|| {
                        datafusion::error::DataFusionError::Execution(
                            "No 'sequence' column found in input".to_string(),
                        )
                    })?;
                
                // Get the sequence column as a StringArray
                let sequence_array = batch
                    .column(sequence_col_idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        datafusion::error::DataFusionError::Execution(
                            "The 'sequence' column is not a string array".to_string(),
                        )
                    })?;
                
                // Calculate base content
                let result_batch = calculate_base_content(sequence_array)?;
                result_batches.push(result_batch);
            }
        }
        
        // Combine results from all partitions
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

        BaseContentExec { input, schema }
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
        Ok(Arc::new(BaseContentExec::new(children[0].clone())))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<PhysicalSendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context.clone())?;
        let schema = self.schema.clone();
        
        let stream = Box::pin(stream! {
            let mut input_stream = input_stream;
            
            while let Some(batch_result) = input_stream.next().await {
                let batch = batch_result?;
                
                // Find the sequence column
                let sequence_col_idx = batch
                    .schema()
                    .fields()
                    .iter()
                    .position(|f| f.name() == "sequence")
                    .ok_or_else(|| {
                        datafusion::error::DataFusionError::Execution(
                            "No 'sequence' column found in input".to_string(),
                        )
                    })?;
                
                // Get the sequence column as a StringArray
                let sequence_array = batch
                    .column(sequence_col_idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        datafusion::error::DataFusionError::Execution(
                            "The 'sequence' column is not a string array".to_string(),
                        )
                    })?;
                
                // Calculate base content
                let result_batch = calculate_base_content(sequence_array)?;
                yield Ok(result_batch);
            }
        });
        
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
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
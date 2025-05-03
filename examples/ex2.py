import polars_bio as pb
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 6)
fastq_file = "tests/data/io/fastq/test.fastq"

df = pb.read_fastq(fastq_file).collect()

df.head()
base_content_df = pb.qc.base_content(df)
print(base_content_df)

pb.qc.visualize_base_content(base_content_df)
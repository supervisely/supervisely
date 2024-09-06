markdown_speedtest_intro = """## Inference Speed

This is a speed test benchmark for this model. The model was tested with the following configuration:

- **Device**: {}
- **Hardware**: {}
- **Runtime**: {}
"""

markdown_speedtest_overview = """
The table below shows the speed test results. For each test, the time taken to process one batch of images is shown, as well as the model's throughput (i.e, the number of images processed per second, or FPS). Results are averaged across **{}** iterations.
"""

markdown_real_time_inference = """## Real-time Inference

This chart compares different runtimes and devices (CPU or GPU)."""

# We additionally divide **predict** procedure into three stages: pre-process, inference, and post-process. Each bar in this chart consists of these three stages. For example, in the chart you can find how long the post-process phase lasts in a CPU device with an ONNXRuntime environment."""


markdown_batch_inference = """
This chart shows how the model's speed changes with different batch sizes . As the batch size increases, you can observe an increase in FPS (images per second).
"""

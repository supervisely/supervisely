markdown_speedtest_intro = """# Inference Speed

We provide a speed test analysis for the model in different scenarios:

1. **Real-time inference** with batch size is set to 1, suitable for processing a stream of images or real-time video capture.
2. **Batch processing**. We assess the scalability of model efficiency with increasing batch size, conducting tests with various batch sizes (i.e, setting batch size to 1, 8, 16).
3. **Runtime environments**. We use both the original PyTorch model and optimized versions exported to runtimes like ONNXRuntime and TensorRT for inference. These optimized runtimes enable efficient model deployment and provide very high throughput.

For more information, please, refer to our <a href="{}" target="_blank">documentation</a>
"""

markdown_speedtest_overview = """## Overview

Speed test results are based on statistics averaged across {} images.

> Note: this benchmark used {} hardware for model inference. Results on different hardware may vary.
"""


markdown_real_time_inference = """## Real-time Inference

This chart compares different runtimes and devices (CPU or GPU)."""

# We additionally divide **predict** procedure into three stages: pre-process, inference, and post-process. Each bar in this chart consists of these three stages. For example, in the chart you can find how long the post-process phase lasts in a CPU device with an ONNXRuntime environment."""


markdown_batch_inference = """## Batch Inference

This chart shows how the model's speed changes with different batch sizes . As the batch size increases, you can observe an increase in FPS (images per second). Each line represents one environment-device setting.
"""

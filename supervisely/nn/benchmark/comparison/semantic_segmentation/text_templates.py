docs_url = (
    "https://docs.supervisely.com/neural-networks/model-evaluation-benchmark/semantic-segmentation"
)

# <i class="zmdi zmdi-check-circle" style="color: #13ce66; margin-right: 5px"></i>
clickable_label = """
> <span style="color: #5a6772">
>     Click on the chart to explore corresponding images.
> </span>
"""

markdown_header = """
<h1>{}</h1>

<div class="model-info-block">
    <div>Created by <b>{}</b></div>
    <div><i class="zmdi zmdi-calendar-alt"></i><span>{}</span></div>
</div>
"""

markdown_common_overview = """
- **Models**: {}
- **Evaluation Dataset**: <a href="/projects/{}/datasets" target="_blank">{}</a>
- **Task type**: {}
"""

markdown_overview_info = """
<h3>{}</h3>
- **Model**: {}
- **Checkpoint**: {}
- **Architecture**: {}
- **Runtime**: {}
- **Checkpoint file**: <a class="checkpoint-url" href="{}" target="_blank">{}</a>
- **Evaluation Report**: <a href="{}" target="_blank">View Report</a>

"""

markdown_key_metrics = """## Key Metrics

We provide a comprehensive analysis of models' performance using a set of metrics, including both basic (precision, recall, F1-score, IoU, etc.) and advanced (boundary IoU, error over union decomposition, etc.) metrics.

- **Pixel accuracy**: reflects the percent of image pixels which were correctly classified.
- **Precision**: reflects the number of correctly predicted positive segmentations divided by the total number of predicted positive segmentations.
- **Recall**: reflects the number of correctly predicted positive segmentations divided by the number of all samples that should have been segmented as positive.
- **F1-score**: reflects the tradeoff between precision and recall. It is equivalent to the Dice coefficient and calculated as a harmonic mean of precision and recall.
- **Intersection over union (IoU, also known as the Jaccard index)**: measures the overlap between ground truth mask and predicted mask. It is calculated as the ratio of the intersection of the two masks areas to their combined areas.
- **Boundary intersection over union**: a segmentation consistency measure that first computes the sets of ground truth  and predicted masks pixels that are located within the distance d from each contour and then computes intersection over union of these two sets. Pixel distance parameter d (pixel width of the boundary region) controls the sensitivity of the metric, it is usually set as 2% of the image diagonal for normal resolution images and 0.5% of the image diagonal for high resolution images.
- **Error over union and its components (boundary, extent, segment)**: a metric opposite to intersection over union and can be interpreted as what the model lacked in order to show the perfect performance with IoU = 1. It reflects the ratio of incorrectly segmented pixels of ground truth and predicted masks to their combined areas. It is usually decomposed into boundary, extent and segment errors over union in order to get exhaustive information about the model's strengths and weaknesses.
- **Renormalized error over union**: postprocessed variant of error over union which takes into consideration cause and effect relationships between different types of segmentation errors.
"""

markdown_explorer = """## Explore Predictions
This section contains visual comparison of predictions made by different models and ground truth annotations. Sometimes a simple visualization can be more informative than any performance metric.

> Click on the image to view the **Original Image** with **Ground Truth** and **Predictions** annotations side-by-side. 
"""

markdown_explore_difference = """## Explore Predictions

In this section, you can explore predictions made by different models side-by-side. This helps you to understand the differences in predictions made by each model, and to identify which model performs better in different scenarios.


> Click on the image to view the **Ground Truth**, and **Prediction** annotations side-by-side.
"""


### Difference in Predictions

# markdown_explore_same_errors = """
# ### Same Errors

# This section helps you to identify samples where all models made the same errors. It is useful for understanding the limitations of the models and the common challenges they face.

# > Click on the image to view the **Ground Truth**, and **Prediction** annotations side-by-side.
# """


# """
markdown_iou = """## Intersection & Error Over Union

Pie charts below demonstrate performance metrics of each model in terms of Intersection over Union (IoU) and Error over Union (EoU). It is done with the help of Error over Union (EoU) decomposition into boundary, extent, and segment errors over union. These charts help to draw conclusions on the model's strongest and weakest sides.
"""

markdown_renormalized_error_ou = """## Renormalized Error over Union

Charts below are dedicated to the decomposition of the post-processed variant of Error over Union, which takes into consideration cause and effect relationships between different types of segmentation errors. Error over Union decomposition has its own pitfalls. It is important to understand that models which tend to produce segment errors (when entire segments are mispredicted and there is no intersection between ground truth and predicted mask) will face fewer occasions to produce boundary and extent errors - as a result, boundary and extent error over union values will be underestimated.

In terms of localization, segment error is more fundamental than extent, while extent error is more fundamental than boundary. In order to overcome this problem, renormalized error over union proposes a slightly different calculation method - by removing more fundamental errors from the denominator - read more in our <a href="{}" target="_blank">technical report</a>
""".format(
    docs_url
)

markdown_eou_per_class = """## Classwise Segmentation Error Analysis

This section contains information about classwise segmentation error decomposition. For each model, each column of the chart represents a certain class from the training dataset, demonstrating model performance in terms of segmenting this specific class on images and what model lacked in order to show the perfect performance.
"""

markdown_frequently_confused_empty = """### Frequently Confused Classes

No frequently confused class pairs found
"""

markdown_frequently_confused = """## Frequently Confused Classes

The bar chart below reveals pairs of classes which were most frequently confused for each model. Each column of the chart demonstrates the probability of confusion of a given pair of classes. It is necessary to remember that this probability is not symmetric: the probability of confusing class A with class B is not equal to the probability of confusing class B with class A.
"""

empty = """### {}

> {}
"""

markdown_speedtest_intro = """## Inference Speed

This is a speed test benchmark for compared models. Models were tested with the following configurations:
"""

markdown_speedtest_overview_ms = """### Latency (Inference Time)
The table below shows the speed test results. For each test, the time taken to process one batch of images is shown. Results are averaged across **{}** iterations.
"""

markdown_speedtest_overview_fps = """### Frames per Second (FPS)
The table below shows the speed test results. For each test, the number of frames processed per second is shown. Results are averaged across **{}** iterations.
"""

markdown_batch_inference = """
This chart shows how the model's speed changes with different batch sizes . As the batch size increases, you can observe an increase in FPS (images per second).
"""

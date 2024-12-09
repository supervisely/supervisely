from types import SimpleNamespace

docs_url = (
    "https://docs.supervisely.com/neural-networks/model-evaluation-benchmark/semantic-segmentation"
)

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


markdown_overview = """
- **Model**: {}
- **Checkpoint**: {}
- **Architecture**: {}
- **Task type**: {}
- **Runtime**: {}
- **Checkpoint file**: <a class="checkpoint-url" href="{}" target="_blank">{}</a>
- **Ground Truth project**: <a href="/projects/{}/datasets" target="_blank">{}</a>, {}{}
{}

Learn more about Model Benchmark, implementation details, and how to use the charts in our <a href="{}" target="_blank">Technical Report</a>.
"""


markdown_key_metrics = """## Key Metrics

We provide a comprehensive model performance analysis using a set of metrics, including both basic (precision, recall, f1-score, IoU, etc.) and advanced (boundary IoU, error over unio decomposition, etc.) metrics.

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
This section contains visual comparison of model predictions and ground truth masks. Sometimes a simple visualization can be more informative than any performance metric.

> Click on the image to view the **Original Image** with **Ground Truth** and **Prediction** annotations side-by-side. 
"""


markdown_predictions_table = """### Prediction details for every image

Table of per image metrics allows to get performance metrics for every image. It can be helpful when there is a need to find the most problematic images where the model performed worst.

**Example**: you can sort by **Pixel accuracy** in ascending order to find images where the model performed worst in terms of pixel-wise accuracy.

> Click on the row to view the **Original Image** with **Ground Truth** and **Prediction** annotations side-by-side. 
"""

markdown_iou = """## Intersection & Error Over Union

The pie chart below demonstrates what the model lacked in order to show the perfect performance with IoU = 1. It is done with the help of Error over Union (EoU) decomposition into boundary, extent and segment errors over union. This chart helps to draw conclusions on the model's strongest and weakest sides.
"""


markdown_renormalized_error_ou = """## Renormalized Error Over Union

The chart below is dedicated to decomposition of postprocessed variant of error over union which takes into consideration cause and effect relationships between different types of segmentation errors. Error over union decomposition has its own pitfalls. It is important to understand that models which tend to produce segment errors (when entire segments are mispredicted and there is no intersection between ground truth and predicted mask) will face less occasions to produce boundary and extent errors - as a result, boundary and extent error over union values will be underestimated.

In terms of localization, segment error is more fundamental than extent, while extent error is more fundamental than boundary. In order to overcome this problem, renormalized error over union proposes a slightly different calculation method - by removing more fundamental errors from the denominator - read more in our <a href="{}" target="_blank">technical report</a>
""".format(
    docs_url
)


markdown_eou_per_class = """## Classwise Segmentation Error Analysis

This section contains information about classwise segmentation error decomposition. Each column of the chart represents a certain class from the training dataset, demonstrating model performance in terms of segmenting this specific class on images and what model lacked in order to show the perfect performance. All classes are sorted in IoU descending order.
"""

markdown_confusion_matrix = """## Confusion Matrix

The confusion matrix below reveals which classes the model commonly confuses with each other.

- **Each row** of the matrix corresponds to the actual instances of a class.
- **Each column** corresponds to the instances as predicted by the model.
- **The diagonal elements** of the matrix represent correctly predicted instances.
- By examining the **off-diagonal elements**, you can see if the model is confusing two classes by frequently mislabeling one as the other.

"""

markdown_frequently_confused = """## Frequently Confused Classes

The bar chart below reveals pairs of classes which were most frequently confused by the model. Each column of the chart demonstrates the probability of confusion of a given pair of classes. It is necessary to remember that this probability is not symmetric: the probability of confusing class A with class B is not equal to the probability of confusing class B with class A.
"""

markdown_frequently_confused_empty = """### Frequently Confused Classes

No frequently confused class pairs found
"""

markdown_speedtest_intro = """## Inference Speed

This is a speed test benchmark for this model. The model was tested with the following configuration:

- **Device**: {}
- **Hardware**: {}
- **Runtime**: {}

The table below shows the speed test results. For each test, the time taken to process one batch of images is shown, as well as the model's throughput (i.e, the number of images processed per second, or FPS). Results are averaged across **{}** iterations.
"""

markdown_batch_inference = """
This chart shows how the model's speed changes with different batch sizes. As the batch size increases, you can observe an increase in FPS (images per second).
"""

markdown_acknowledgement = """---
### Acknowledgement

[1] Maximilian Bernhard, Roberto Amoroso, Yannic Kindermann, Lorenzo Baraldi, Rita Cucchiara, Volker Tresp, Matthias Schubert. <a href="https://openaccess.thecvf.com/content/WACV2024/html/Bernhard_Whats_Outside_the_Intersection_Fine-Grained_Error_Analysis_for_Semantic_Segmentation_WACV_2024_paper.html" target="_blank">What's Outside the Intersection? Fine-grained Error Analysis for Semantic Segmentation Beyond IoU.</a> In Proceedings of the IEEE / CVF Conference on Computer Vision and Pattern Recognition, pages 969 - 977, 2024.

[2] Bowen Cheng, Ross Girshick, Piotr Dollar, Alexander C. Berg, Alexander Kirillov. <a href="https://arxiv.org/abs/2103.16562" target="_blank">Boundary IoU: Improving object-centric image segmentation evaluation.</a> In Proceedings of the IEEE / CVF Conference on Computer Vision and Pattern Recognition, pages 15334 - 15342, 2021.
"""

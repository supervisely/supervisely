from types import SimpleNamespace

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

Table of per image metrics allows to get performance metrics for every image. It can be helpful when there is a need to find the most problematic images where model performed worst.

> Click on the row to view the image with **Ground Truth** and **Prediction** annotations.
"""

pixel_accuracy = """## Pixel accuracy

Pixel accuracy is calculated as a sum of correctly segmented image pixels (TP and TN) divided by total number of image pixels. Pixel accuracy is an easy-to-understand metric, but it can be misleading in conditions of class imbalance. 

Pixel accuracy provides superficial understanding of model performance, but it does not give information about where exactly the model makes mistakes (which error type is prevailing - false positive or false negative?). Nevertheless, pixel accuracy is still an inalienable part of any semantic segmentation performance evaluation.

"""

markdown_precision = """## Precision

Precision is calculated as a number correctly segmented positive image pixels (where target class is present) divided by total number segmented image pixels (both correctly as TP and incorrectly as FP). 

Precision measures the accuracy of positive predictions and can be useful in cases when the cost of false positive errors is high (for example, spam filtering) and it is important to minimize them. At the same time precision does not take into account false negative errors. So this metric alone will not provide a complete picture of neural network performance, it is better to use precision in combination with other evaluation metrics in order to get unbiased information about model performance.

"""

notification_precision = {
    "title": "Precision = {}",
    "description": "The model correctly predicted <b>{} of {}</b> predictions made by the model in total.",
}


markdown_recall = """## Recall

Recall is calculated as a number correctly segmented positive image pixels (where target class is present) divided by total number image pixels that must be segmented (sum of TP and FN). Recall measures the model's ability to correctly segment all positive instances and can be useful in cases when it is necessary to minimize false negative errors (for example, disease diagnostics). 

Disadvantages of this metric are similar to disadvantages of precision: recall does not take into account false positive errors - it means that recall will be not representative on significantly imbalanced data (since the model which segments all pixels as positive will have high recall, but very low precision). So general recommendation for recall usage is the same as for precision - use it in combination with other evaluation metrics.

"""

notification_recall = {
    "title": "Recall = {}",
    "description": "The model correctly found <b>{} of {}</b> total instances in the dataset.",
}


markdown_f1_score = """## F1-score

F1-score is calculated as a harmonic mean of precision and recall. F1-score combines precision and recall into a single evaluation metric. It is especially useful in cases when there is a necessity in minimizing both false positive and false negative errors (it is highly demanded in the medical imaging domain). 

On the other hand, it is necessary to remember that this metric sets equal weights to precision and recall, which might be not suitable for cases when the cost of false positive and false negative error is not equal.
"""

markdown_iou = """## Intersection & Error Over Union (IoU)

### Intersection Over Union (IoU)

Intersection over union is calculated as number of image pixels located at the intersection of the ground truth and predicted masks areas divided by number of image pixels located at combined area of ground truth and predicted masks. 

Intersection over union is currently being used as a gold standard for comparing semantic segmentation models.

### Error Over Union and its components: boundary, extent, segment

The following segmentation error analysis methodology was proposed by Maximilian Bernhard, Roberto Amoroso, Yannic Kindermann, Lorenzo Baraldi, Rita Cucchiara, Volker Tresp and Matthias Schubert in their paper “What's Outside the Intersection? Fine-grained Error Analysis for Semantic Segmentation Beyond IoU” [1].

Error over union is calculated as number of incorrectly segmented image pixels (as a sum of FP and FN) divided by number of image pixels located at combined area of ground truth and predicted masks. Error over union is a metric opposite to intersection over union and can be interpreted as what the model lacked in order to show the perfect performance with IoU = 1. It is usually decomposed into boundary, extent and segment errors over union.

Semantic segmentation errors can be divided into 3 categories: boundary, segment and extent.

Boundary error occurs when  a transition between foreground and background for a class has been recognized, but not delineated perfectly. Extent errors occur when a segment has been recognized, but under- or overestimated in its extent.

Extent errors occur when a segment has been recognized, but under- or overestimated in its extent. False positive extent errors are pixels that belong to a contiguous predicted segment which intersects with the ground truth foreground. False negative extent errors are pixels that belong to a contiguous ground-truth segment which intersects with the predicted foreground.

Segment errors have no apparent relation to true positive predictions. False positive segment errors are predicted segments that do not have any intersection with the ground truth foreground. False negative segment errors are ground truth foreground segments that do not have any intersection with the predicted foreground.

### Boundary Intersection Over Union

This metric was proposed by Bowen Cheng, Ross Girshick, Piotr Dollar, Alexander C. Berg and Alexander Kirillov in their article “Boundary IoU: Improving object-centric image segmentation evaluation” [2].

Boundary IoU is a segmentation consistency measure that first computes the sets of ground truth (G)  and predicted (P)  masks’ pixels that are located within the distance d (2% of the image diagonal) from each contour (Gd and Pd respectively)  and then computes intersection over union of these two sets. It is necessary to remember that intersection over union values all pixels equally and, therefore, is less sensitive to boundary quality in larger objects: the number of interior pixels grows quadratically in object size and can far exceed the number of boundary pixels, which only grows linearly. Boundary IoU allows to overcome this limitation and objectively estimate boundary segmentation quality across all scales.
"""


markdown_renormalized_error_ou = """## Renormalized Error Over Union

Error over union decomposition has its own pitfalls. It is important to understand that models which tend to produce segment errors (when entire segments are mispredicted and there is no intersection between ground truth and predicted mask) will face less occasions to produce boundary and extent errors - as a result, boundary and extent error over union values will be underestimated.

In terms of localization, segment error is more fundamental than extent, while extent error is more fundamental than boundary. In order to overcome this problem, renormalized error over union proposes a slightly different calculation method - by removing more fundamental errors from the denominator - read more in our technical report.
"""


markdown_eou_per_class = """## Classwise Segmentation Error Analysis

This chart displays the segmentation error analysis for each class. It shows the distribution of boundary, extent, and segment errors over union values for each class.
"""

markdown_confusion_matrix = """## Confusion Matrix

The confusion matrix reveals which classes the model commonly confuses with each other.

- **Each row** of the matrix corresponds to the actual instances of a class.
- **Each column** corresponds to the instances as predicted by the model.
- **The diagonal elements** of the matrix represent correctly predicted instances.
- By examining the **off-diagonal elements**, you can see if the model is confusing two classes by frequently mislabeling one as the other.

"""

markdown_frequently_confused = """## Frequently Confused Classes

This chart displays the most frequently confused pairs of classes. In general, it finds out which classes visually seem very similar to the model. It calculates the **probability of confusion** between different pairs of classes.
"""

markdown_speedtest_intro = """## Inference Speed

This is a speed test benchmark to evaluate the model's performance in terms of inference time. The model was tested with the following configurations:
"""

markdown_speedtest_overview = """### Latency (Inference Time)
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

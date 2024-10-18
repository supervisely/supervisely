from types import SimpleNamespace

definitions = SimpleNamespace(
    true_positives="True Positive (TP) is a correctly detected object. For a prediction to be counted as a true positive, the predicted segmentation mask must align with a ground truth segmentation mask with IoU of 0.5 or more, and the object must be correctly classified.",
    false_positives="False Positive (FP) occurs when the model predicts an object that is not actually present in the image. For example, the model predicted a car, but no car is annotated. A false positive detection happens when a ground truth counterpart is not found for a prediction (i.e, IoU of the predicted mask is less than 0.5 with any ground truth mask), or the model predicted incorrect class for an object.",
    false_negatives="False Negative (FN) happens when the model fails to detect an object that is present in the image. For example, the model does not detect a car that is actually in the ground truth. A false negative detection occurs when a ground truth segmentation mask has no any predicted mask with IoU greater than 0.5, or their classes do not match.",
    confidence_threshold="Confidence threshold is a hyperparameter used to filter out predictions that the model is not confident in. By setting a higher confidence threshold, you ensure that only the most certain predictions are considered, thereby reducing the number of false predictions. This helps to control the trade-off between precision and recall in the model's output.",
    confidence_score="The confidence score, also known as probability score, quantifies how confident the model is that its prediction is correct. It is a numerical value between 0 and 1, generated by the model for each predicted object, that represents the likelihood that the prediction contains an object of a particular class.",
    f1_score="F1-score is a useful metric that combines both precision and recall into a single measure. As the harmonic mean of precision and recall, the f1-score provides a balanced representation of both metrics in one value. F1-score ranges from 0 to 1, with a higher score indicating better model performance. It is calculated as 2 * (precision * recall) / (precision + recall).",
    average_precision="Average precision (AP) is computed as the area under the precision-recall curve. It measures the precision of the model at different recall levels and provides a single number that summarizes the trade-off between precision and recall for a given class.",
    about_pr_tradeoffs="A system with high recall but low precision returns many results, but most of its predictions are incorrect or redundant (false positive). A system with high precision but low recall is just the opposite, returning very few results, most of its predictions are correct. An ideal system with high precision and high recall will return many results, with all results predicted correctly.",
    iou_score="Intersection over Union (IoU) measures the overlap between two masks: one predicted by the model and one from the ground truth. It is calculated as the area of intersection between the predicted mask and the ground truth mask, divided by the area of their union. A higher IoU score indicates better alignment between the predicted and ground truth masks.",
    iou_threshold="The IoU threshold is a predefined value (set to 0.5 in many benchmarks) that determines the minimum acceptable IoU score for a predicted mask to be considered a correct prediction. When the IoU of a predicted mask and actual mask is higher than this IoU threshold, the prediction is considered correct. Some metrics will evaluate the model with different IoU thresholds to provide more insights about the model's performance.",
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
- **Checkpoint file**: <a href="{}" target="_blank">{}</a>
- **Ground Truth project**: <a href="/projects/{}/datasets" target="_blank">{}</a>, {}{}
{}

Learn more about Model Benchmark, implementation details, and how to use the charts in our <a href="{}" target="_blank">Technical Report</a>.
"""
# - **Model**: {}
# - **Training dataset (?)**: COCO 2017 train
# - **Model classes (?)**: (80): a, b, c, … (collapse)
# - **Model weights (?)**: [/path/to/yolov8l.pt]()
# - **License (?)**: AGPL-3.0

markdown_key_metrics = """## Key Metrics

Here, we comprehensively assess the model's performance by presenting a broad set of metrics, including mAP (mean Average Precision), Precision, Recall, IoU (Intersection over Union), Classification Accuracy and Calibration Score.

- **Mean Average Precision (mAP)**: A comprehensive metric of detection and instance segmentation performance. mAP calculates the <abbr title="{}">average precision</abbr> across all classes at different levels of <abbr title="{}">IoU thresholds</abbr> and precision-recall trade-offs. In other words, it evaluates the performance of a model by considering its ability to detect and localize objects accurately across multiple IoU thresholds and object categories.
- **Precision**: Precision indicates how often the model's predictions are actually correct when it predicts an object. This calculates the ratio of correct predictions to the total number of predictions made by the model.
- **Recall**: Recall measures the model's ability to find all relevant objects in a dataset. This calculates the ratio of correct predictions to the total number of instances in a dataset.
- **Intersection over Union (IoU)**: IoU measures the overlap between two masks: one predicted by the model and one from the ground truth. It is calculated as the area of intersection between the predicted mask and the ground truth mask, divided by the area of their union. A higher IoU score indicates better alignment between the predicted and ground truth masks.
- **Classification Accuracy**: We additionally measure the classification accuracy of an instance segmentation model. This metric represents the percentage of correctly labeled instances among all instances where the predicted segmentation masks accurately match the ground truth masks (with an IoU greater than 0.5, regardless of class).
- **Calibration Score**: This score represents the consistency of predicted probabilities (or <abbr title="{}">confidence scores</abbr>) made by the model. We evaluate how well predicted probabilities align with actual outcomes. A well-calibrated model means that when it predicts an object with, say, 80% confidence, approximately 80% of those predictions should actually be correct.
"""

markdown_explorer = """## Explore Predictions
In this section you can visually assess the model performance through examples. This helps users better understand model capabilities and limitations, giving an intuitive grasp of prediction quality in different scenarios.

> Click on the image to view the **Ground Truth**, **Prediction**, and **Difference** annotations side-by-side. 

> Filtering options allow you to adjust the confidence threshold (only for predictions) and the model's false outcomes (only for differences). Differences are calculated only for the optimal confidence threshold, allowing you to focus on the most accurate predictions made by the model.
"""

markdown_predictions_gallery = """

"""
# You can choose one of the sorting method:

# - **Auto**: The algorithm is trying to gather a diverse set of images that illustrate the model's performance across various scenarios.
# - **Least accurate**: Displays images where the model made more errors.
# - **Most accurate**: Displays images where the model made fewer or no errors.
# - **Dataset order**: Displays images in the original order of the dataset.
# """

markdown_predictions_table = """### Prediction details for every image

The table helps you in finding samples with specific cases of interest. You can sort by parameters such as the number of predictions, or specific a metric, e.g, recall, then click on a row to view this image and predictions.

**Example**: you can sort by **FN** (False Negatives) in descending order to identify samples where the model failed to detect many objects.


> Click on the row to view the image with **Ground Truth**, **Prediction**, or the **Difference** annotations.
"""

markdown_what_is = """
"""

markdown_experts = """
"""

markdown_how_to_use = """
"""

markdown_outcome_counts = (
    """## Outcome Counts

This chart is used to evaluate the overall model performance by breaking down all predictions into <abbr title="{}">True Positives</abbr> (TP), <abbr title="{}">False Positives</abbr> (FP), and <abbr title="{}">False Negatives</abbr> (FN). This helps to visually assess the type of errors the model often encounters.

"""
    + clickable_label
)

markdown_R = """## Recall

This section measures the ability of the model to find **all relevant instances in the dataset**. In other words, it answers the question: “Of all instances in the dataset, how many of them is the model managed to find out?”

To measure this, we calculate **Recall**. Recall counts errors, when the model does not predict an object that actually is present in a dataset and should be predicted. Recall is calculated as the portion of correct predictions (true positives) over all instances in the dataset (true positives + false negatives).
"""

notification_recall = {
    "title": "Recall = {}",
    "description": "The model correctly found <b>{} of {}</b> total instances in the dataset.",
}

markdown_R_perclass = (
    """### Per-class Recall

This chart further analyzes Recall, breaking it down to each class in separate.

Since the overall recall is calculated as an average across all classes, we provide a chart showing the recall for each individual class. This illustrates how much each class contributes to the overall recall.

_Bars in the chart are sorted by <abbr title="{}">F1-score</abbr> to keep a unified order of classes between different charts._

"""
    + clickable_label
)


markdown_P = """## Precision

This section measures the accuracy of all predictions made by the model. In other words, it answers the question: “Of all predictions made by the model, how many of them are actually correct?”.

To measure this, we calculate **Precision**. Precision counts errors, when the model predicts an object, but the image has no objects of the predicted class in this place. Precision is calculated as a portion of correct predictions (true positives) over all model’s predictions (true positives + false positives).
"""

notification_precision = {
    "title": "Precision = {}",
    "description": "The model correctly predicted <b>{} of {}</b> predictions made by the model in total.",
}

markdown_P_perclass = (
    """### Per-class Precision

This chart further analyzes Precision, breaking it down to each class in separate.

Since the overall precision is computed as an average across all classes, we provide a chart showing the precision for each class individually. This illustrates how much each class contributes to the overall precision.

_Bars in the chart are sorted by <abbr title="{}">F1-score</abbr> to keep a unified order of classes between different charts._

"""
    + clickable_label
)


markdown_PR = (
    """## Recall vs. Precision

This section compares Precision and Recall in one graph, identifying **imbalance** between these two.

_Bars in the chart are sorted by <abbr title="{}">F1-score</abbr> to keep a unified order of classes between different charts._

"""
    + clickable_label
)


markdown_pr_curve = """## Precision-Recall Curve

Precision-Recall curve is an overall performance indicator. It helps to visually assess both precision and recall for all predictions made by the model on the whole dataset. This gives you an understanding of how precision changes as you attempt to increase recall, providing a view of **trade-offs between precision and recall** <abbr title="{}">(?)</abbr>. Ideally, a high-quality model will maintain strong precision as recall increases. This means that as you move from left to right on the curve, there should not be a significant drop in precision. Such a model is capable of finding many relevant instances, maintaining a high level of precision.
"""

markdown_trade_offs = """- A system with high recall but low precision generates many results, but most of its predictions are incorrect or redundant (false positives).

- Conversely, a system with high precision but low recall produces very few results, but most of its predictions are accurate.

- The ideal system achieves both high precision and high recall, meaning it returns many results with a high accuracy rate.
"""

markdown_what_is_pr_curve = """1. **Sort predictions**: Arrange all predicted objects by their <abbr title="{}">confidence scores</abbr> in descending order.

2. **Classify outcomes**: For each prediction, determine if it is a <abbr title="{}">true positive</abbr> (TP) or a <abbr title="{}">false positive</abbr> (FP) and record these classifications in a table.

3. **Calculate cumulative metrics**: As you move through each prediction, calculate the cumulative precision and recall. Add these values to the table.

4. **Plot points**: Each row in the table now represents a point on a graph, with cumulative recall on the x-axis and cumulative precision on the y-axis. Initially, this creates a zig-zag line because of variations between predictions.

5. **Smooth the curve**: The true PR curve is derived by plotting only the maximum precision value for each recall level across all thresholds. This means you connect only the highest points of precision for each segment of recall, smoothing out the zig-zags and forming a curve that typically slopes downward as recall increases.
"""


notification_ap = {
    "title": "mAP = {}",
    "description": "",
}

markdown_pr_by_class = (
    """### Precision-Recall Curve by Class

In this plot, you can evaluate PR curve for each class individually.

"""
    + clickable_label
)

markdown_confusion_matrix = (
    """## Confusion Matrix

Confusion matrix helps to find the number of confusions between different classes made by the model.
Each row of the matrix represents the instances in a ground truth class, while each column represents the instances in a predicted class.
The diagonal elements represent the number of correct predictions for each class (True Positives), and the off-diagonal elements show misclassifications.

"""
    + clickable_label
)


markdown_frequently_confused = (
    """### Frequently Confused Classes

This chart displays the most frequently confused pairs of classes. In general, it finds out which classes visually seem very similar to the model.

The chart calculates the **probability of confusion** between different pairs of classes. For instance, if the probability of confusion for the pair “{} - {}” is {}, this means that when the model predicts either “{}” or “{}”, there is a {}% chance that the model might mistakenly predict one instead of the other.

The measure is class-symmetric, meaning that the probability of confusing a {} with a {} is equal to the probability of confusing a {} with a {}.

"""
    + clickable_label
)


markdown_localization_accuracy = """## Mask accuracy (IoU)

This section measures how accurately predicted masks match the actual shapes of ground truth instances. We calculate the average <abbr title="{}">IoU score</abbr> of predictions and visualize a histogram of IoU scores.
"""

markdown_iou_calculation = """<img src='https://github.com/dataset-ninja/model-benchmark-template/assets/78355358/8d7c63d0-2f3b-4f3f-9fd8-c6383a4bfba4' alt='alt text' width='300' />

To measure it, we calculate the <b>Intersection over Union (IoU)</b>. IoU measures the overlap between two masks: one predicted by the model and one from the ground truth. Unlike object detection, which uses rectangular bounding boxes, instance segmentation deals with masks that represent the exact shape of an object. It is calculated similarly by taking the area of intersection between the predicted mask and the ground truth mask and dividing it by the area of their union.
"""

markdown_iou_distribution = """### IoU Distribution

This histogram represents the distribution of <abbr title="{}">IoU scores</abbr> among all predictions. This gives a sense of how accurate the model is in generating masks of the objects. Ideally, the rightmost bars (from 0.9 to 1.0 IoU) should be much higher than others.
"""


notification_avg_iou = {
    "title": "Avg. IoU = {}",
    "description": "",
}

markdown_calibration_score_1 = """## Calibration Score

This section analyzes <abbr title="{}">confidence scores</abbr> (or predicted probabilities) that the model generates for every predicted segmentation mask.
"""

markdown_what_is_calibration = """In some applications, it's crucial for a model not only to make accurate predictions but also to provide reliable **confidence levels**. A well-calibrated model aligns its confidence scores with the actual likelihood of predictions being correct. For example, if a model claims 90% confidence for predictions but they are correct only half the time, it is **overconfident**. Conversely, **underconfidence** occurs when a model assigns lower confidence scores than the actual likelihood of its predictions. In the context of autonomous driving, this might cause a vehicle to brake or slow down too frequently, reducing travel efficiency and potentially causing traffic issues."""
markdown_calibration_score_2 = """To evaluate the calibration, we draw a <b>Reliability Diagram</b> and calculate <b>Expected Calibration Error</b> (ECE)."""

markdown_reliability_diagram = """### Reliability Diagram

Reliability diagram, also known as a Calibration curve, helps in understanding whether the confidence scores of a model accurately represent the true probability of a correct prediction. A well-calibrated model means that when it predicts an instance with, say, 80% confidence, approximately 80% of those predictions should actually be correct.
"""

markdown_calibration_curve_interpretation = """
1. **The curve is above the perfect line (Underconfidence):** If the calibration curve is consistently above the perfect line, this indicates underconfidence. The model’s predictions are more correct than the confidence scores suggest. For example, if the model assigns 70% confidence to some predictions but, empirically, 90% of these predictions are correct, the model is underconfident.
2. **The curve is below the perfect line (Overconfidence):** If the calibration curve is below the perfect line, the model exhibits overconfidence. This means it is too sure of its predictions. For example, if the model assigns 80% confidence to some predictions, but only 40% of these predictions are correct, the model is overconfident.

To quantify the calibration, we calculate **Expected Calibration Error (ECE).** Intuitively, ECE can be viewed as a deviation of the model's calibration curve from the diagonal line, that corresponds to a perfectly calibrated model. When ECE is high, we can not trust predicted probabilities so much.

**Note:** ECE is a measure of **error**. The lower the ECE, the better the calibration. A perfectly calibrated model has an ECE of 0.
"""

notification_ece = {
    "title": "Expected Calibration Error (ECE) = {}",
    "description": "",
}


markdown_confidence_score_1 = """## Confidence Score Profile

This section is going deeper in analyzing confidence scores. It gives you an intuition about how these scores are distributed and helps to find the best <abbr title="{}">confidence threshold</abbr> suitable for your task or application.
"""

markdown_confidence_score_2 = """This chart provides a comprehensive view about predicted confidence scores. It is used to determine an **optimal confidence threshold** based on your requirements.

The plot shows you what the metrics will be if you choose a specific confidence threshold. For example, if you set the threshold to 0.32, you can see on the plot what the precision, recall and f1-score will be for this threshold.
"""

markdown_plot_confidence_profile = """
First, we sort all predictions by confidence scores from highest to lowest. As we iterate over each prediction we calculate the cumulative precision, recall and f1-score so far. Each prediction is plotted as a point on a graph, with a confidence score on the x-axis and one of three metrics on the y-axis (precision, recall, f1-score).
"""

markdown_calibration_score_3 = """**How to find an optimal threshold:** you can find the maximum of the f1-score line on the plot, and the confidence score (X-axis) under this maximum corresponds to F1-optimal confidence threshold. This threshold ensures the balance between precision and recall. You can select a threshold according to your desired trade-offs."""

notification_f1 = {
    "title": "F1-optimal confidence threshold = {}",
    "description": "",
}

markdown_f1_at_ious = """### Confidence Profile at Different IoU thresholds

This chart breaks down the Confidence Profile into multiple curves, each for one <abbr title="{}">IoU threshold</abbr>. In this way you can understand how the f1-optimal confidence threshold changes with various IoU thresholds. Higher IoU thresholds mean that the model should generate more accurate masks to get a correct prediction. This chart helps to find the optimal confidence threshold for different levels of mask accuracy.
"""
markdown_confidence_distribution = """### Confidence Distribution
This graph helps to assess whether high confidence scores correlate with correct predictions (<abbr title="{}">true positives</abbr>) and the low confidence scores correlate with incorrect ones (<abbr title="{}">false positives</abbr>). It consists of two histograms, one for true positive predictions filled with green, and one for false positives filled with red.

Additionally, it provides a view of how predicted probabilities are distributed. Whether the model skews probabilities to lower or higher values, leading to imbalance?

Ideally, the green histogram (TP predictions) should have higher confidence scores and be shifted to the right, indicating that the model is sure about its correct predictions, and the red histogram (FP predictions) should have lower confidence scores and be shifted to the left.
"""

markdown_class_ap = (
    """## Average Precision by Class

A quick visual comparison of the model performance across all classes. Each axis in the chart represents a different class, and the distance to the center indicates the <abbr title="{}">Average Precision</abbr> (AP) for that class.

"""
    + clickable_label
)


markdown_class_outcome_counts_1 = """### Outcome Counts by Class

This chart breaks down all predictions into <abbr title="{}">True Positives</abbr> (TP), <abbr title="{}">False Positives</abbr> (FP), and <abbr title="{}">False Negatives</abbr> (FN) by classes. This helps to visually assess the type of errors the model often encounters for each class.

"""

markdown_normalization = """Normalization is used for better interclass comparison. If the normalization is on, the total outcome counts are divided by the number of ground truth instances of the corresponding class. This is useful, because on the chart, the sum of TP and FN bars will always result in 1.0, representing the full set of ground truth instances in the dataset for a class. This provides a clear visual understanding of how many instances the model correctly detected, how many it missed, and how many were false positives. For example, if a green bar (TP outcomes) reaches 1.0, this means the model has managed to predict all objects for the class without false negatives. Everything that is higher than 1.0 corresponds to False Positives, i.e, redundant predictions that the model should not predict. You can turn off the normalization, switching to absolute counts.

If normalization is off, the chart will display the total count of instances that correspond to outcome type (one of TP, FP or FN). This mode is identical to the main Outcome Counts graph on the top of the page. However, when normalization is off, you may encounter a class imbalance problem. Visually, bars that correspond to classes with many instances in the dataset will be much larger than others. This complicates the visual analysis.
"""

markdown_class_outcome_counts_2 = (
    """You can switch the plot view between normalized and absolute values.

_Bars in the chart are sorted by <abbr title="{}">F1-score</abbr> to keep a unified order of classes between different charts._

"""
    + clickable_label
)

empty = """### {}

> {}
"""

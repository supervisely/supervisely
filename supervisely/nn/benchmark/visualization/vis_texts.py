from types import SimpleNamespace

definitions = SimpleNamespace(
    true_positives="True Positives (TP): These are correctly detected objects. For a prediction to be counted as a true positive, the predicted bounding box must align with a ground truth bounding box with an Intersection over Union (IoU) of 0.5 or more, and the object must be correctly classified",
    false_positives="False Positives (FP): These are incorrect detections made by the model. They occur when the model predicts a bounding box that either does not overlap sufficiently with any ground truth box (IoU less than 0.5) or incorrectly classifies the object within the bounding box. For example, the model detects a car in the image, but there is no car in the ground truth.",
    false_negatives="False Negatives (FN): These are the missed detections. They occur when an actual object in the ground truth is not detected by the model, meaning there is no predicted bounding box with an IoU of 0.5 or more for this object. For example, there is a car in the image, but the model fails to detect it.",
    confidence_threshold="Confidence threshold is a hyperparameter used to filter out predictions that the model is not confident in. It helps to control the trade-off between precision and recall in the model's output. By setting a higher confidence threshold, you ensure that only the most certain predictions are considered, thereby reducing the number of false predictions.",
    confidence_score="The confidence score, also known as probability score, quantifies how confident the model is that its prediction is correct. It is a numerical value between 0 and 1, generated by the model for each bounding box, that represents the likelihood that a predicted bounding box contains an object of a particular class.",
    f1_score="F1 Score is the harmonic mean of precision and recall. It is a useful metric when you need to balance precision and recall. It is calculated as 2 * (precision * recall) / (precision + recall).",
    average_precision="Average precision (AP) is computed as the area under the precision-recall curve. It measures the precision of the model at different recall levels and provides a single number that summarizes the trade-off between precision and recall for a given class.",
    about_pr_tradeoffs="A system with high recall but low precision returns many results, but most of its predictions are incorrect or redundant (false positive). A system with high precision but low recall is just the opposite, returning very few results, most of its predictions are correct. An ideal system with high precision and high recall will return many results, with all results predicted correctly.",
    iou_score="IoU score is a measure of overlap between predicted bounding box and ground truth bounding box. A higher IoU score indicates better alignment between the predicted and ground truth bounding boxes.",
    iou_threshold="The IoU threshold is a predefined value (set at 0.5 in many benchmarks) that determines the minimum acceptable IoU score for a predicted bounding box to be considered a correct detection. When the IoU of a predicted bounding box and an actual bounding box is below this threshold, the prediction is considered a false positive. Higher IoU thresholds require more precise localization, which can lead to lower metrics if the model's predictions are less accurate.",
)

checkpoint_name = "YOLOv8-L (COCO 2017 val)"

# <i class="zmdi zmdi-check-circle" style="color: #13ce66; margin-right: 5px"></i>
clickable_label = """
<span style="color: #5a6772">
    Click on the chart to explore corresponding images.
</span>
"""

markdown_overview = """# {}

## Overview

- **Architecture**: {}
- **Task type**: {}
- **Runtime**: {}
- **Hardware**: {}
- **Checkpoint URL**: <a href="{}" target="_blank">{}</a>
- Learn metrics details and how to use them in <a href="{}" target="_blank">technical report</a>
"""
# - **Model**: {}
# - **Training dataset (?)**: COCO 2017 train
# - **Model classes (?)**: (80): a, b, c, … (collapse)
# - **Model weights (?)**: [/path/to/yolov8l.pt]()
# - **License (?)**: AGPL-3.0

markdown_key_metrics = """## Key Metrics

Here, we comprehensively assess the model's performance by presenting a broad set of metrics, including mAP (mean Average Precision), Precision, Recall, IoU (Intersection over Union), Classification Accuracy, Calibration Score, and Inference Speed.

- **Mean Average Precision (mAP)**: A comprehensive metric of detection performance. mAP calculates the <abbr title="{}">average precision</abbr> across all classes at different levels of <abbr title="{}">IoU thresholds</abbr> and precision-recall trade-offs. In other words, it evaluates the performance of a model by considering its ability to detect and localize objects accurately across multiple IoU thresholds and object categories.
- **Precision**: Precision indicates how often the model's predictions are actually correct when it predicts an object. This calculates the ratio of correct detections to the total number of detections made by the model.
- **Recall**: Recall measures the model's ability to find all relevant objects in a dataset. This calculates the ratio of correct detections to the total number of instances in a dataset.
- **Intersection over Union (IoU)**: IoU measures how closely predicted bounding boxes match the actual (ground truth) bounding boxes. It is calculated as the area of overlap between the predicted bounding box and the ground truth bounding box, divided by the area of union of these bounding boxes.
- **Classification Accuracy**: We separately measure the model's capability to correctly classify objects. It's calculated as a proportion of correctly classified objects among all matched detections. A predicted bounding box is considered matched if it overlaps a ground true bounding box with IoU equal or higher than 0.5.
- **Calibration Score**: This score represents the consistency of predicted probabilities (or <abbr title="{}">confidence scores</abbr>) made by the model, evaluating how well the predicted probabilities align with actual outcomes. A well-calibrated model means that when it predicts a detection with, say, 80% confidence, approximately 80% of those predictions should actually be correct.
- **Inference Speed**: The number of frames per second (FPS) the model can process, measured with a batch size of 1. The inference speed is important in applications, where real-time object detection is required. Additionally, slower models pour more GPU resources, so their inference cost is higher.
"""

markdown_explorer = """## Explore Predictions
In this section you can visually assess the model performance through examples. This helps users better understand model capabilities and limitations, giving an intuitive grasp of prediction quality in different scenarios.

Explore the model's predictions on the grid. Click one of the images to view the **Ground Truth**, **Prediction**, or the **Difference** annotations on the side-by-side view. This helps you to recognize mistakes and peculiarities of the data, which will be shown in the **Difference** column. The *filter* option allows you to change the *confidence* threshold and model's false *outcomes*.


> Note that in the modal with the difference views the **threshold** filter is applied only to the **Prediction** column, while the **outcome** filter is applied to the **Difference** column. 
The **Difference** is calculated only for the optimal confidence threshold, which is the confidence score that maximizes the F1-score. This allows you to focus on the most accurate predictions made by the model.
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
\n\n*Click on the row* to view the image with **Ground Truth**, **Prediction**, or the **Difference** annotations.
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

This section measures the ability of the model to detect **all relevant instances in the dataset**. In other words, this answers the question: “Of all instances in the dataset, how many of them is the model managed to find out?”

To measure this, we calculate **Recall**. Recall counts errors, when the model does not detect an object that actually is present in a dataset and should be detected. Recall is calculated as the portion of correct predictions (true positives) over all instances in the dataset (true positives + false negatives).
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

This section measures the accuracy of all predictions made by the model. In other words, this answers the question: “Of all predictions made by the model, how many of them are actually correct?”.

To measure this, we calculate **Precision**. Precision counts errors, when the model predicts an object (bounding box), but the image has no objects of the predicted class in this place. Precision is calculated as a portion of correct predictions (true positives) over all model’s predictions (true positives + false positives).
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


markdown_PR = """## Recall vs. Precision

This section compares Precision and Recall on a common graph, identifying **disbalance** between these two.

_Bars in the chart are sorted by <abbr title="{}">F1-score</abbr> to keep a unified order of classes between different charts._

<i class="zmdi zmdi-check-circle" style="color: #13ce66; margin-right: 5px"></i>
  <span style="color: #5a6772">
    Click on the chart to explore corresponding images.
  </span>
</div>
"""


markdown_pr_curve = """## Precision-Recall Curve

Precision-Recall curve is an overall performance indicator. It helps to visually assess both precision and recall for all predictions made by the model on the whole dataset. This gives you an understanding of how precision changes as you attempt to increase recall, providing a view of **trade-offs between precision and recall** <abbr title="{}">(?)</abbr>. Ideally, a high-quality model will maintain strong precision as recall increases. This means that as you move from left to right on the curve, there should not be a significant drop in precision. Such a model is capable of finding many relevant instances, maintaining a high level of precision.
"""

markdown_trade_offs = """A system with high recall but low precision returns many results, but most of its predictions are incorrect or redundant (false positive). A system with high precision but low recall is just the opposite, returning very few results, most of its predictions are correct. An ideal system with high precision and high recall will return many results, with all results predicted correctly."""

markdown_what_is_pr_curve = """Imagine you sort all the predictions by their <abbr title="{}">confidence scores</abbr> from highest to lowest and write it down in a table. As you iterate over each sorted prediction, you classify it as a <abbr title="{}">true positive</abbr> (TP) or a <abbr title="{}">false positive</abbr> (FP). For each prediction, you then calculate the cumulative precision and recall so far. Each prediction is plotted as a point on a graph, with recall on the x-axis and precision on the y-axis. Now you have a plot very similar to the PR-curve, but it appears as a zig-zag curve due to variations as you move from one prediction to the next.

 **Forming the Actual PR Curve**: The true PR curve is derived by plotting only the maximum precision value for each recall level across all thresholds.
This means you connect only the highest points of precision for each segment of recall, smoothing out the zig-zags and forming a curve that typically slopes downward as recall increases.
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

markdown_confusion_matrix = """## Confusion Matrix

Confusion matrix helps to find the number of confusions between different classes made by the model.
Each row of the matrix represents the instances in a ground truth class, while each column represents the instances in a predicted class.
The diagonal elements represent the number of correct predictions for each class (True Positives), and the off-diagonal elements show misclassifications.

*Click on the chart to explore corresponding images.*
"""


markdown_frequently_confused = (
    """### Frequently Confused Classes

This chart displays the most frequently confused pairs of classes. In general, it finds out which classes visually seem very similar to the model.

The chart calculates the **probability of confusion** between different pairs of classes. For instance, if the probability of confusion for the pair “{} - {}” is {}, this means that when the model predicts either “{}” or “{}”, there is a {}% chance that the model might mistakenly predict one instead of the other.

The measure is class-symmetric, meaning that the probability of confusing a {} with a {} is equal to the probability of confusing a {} with a {}.

"""
    + clickable_label
)


markdown_localization_accuracy = """## Localization Accuracy (IoU)

This section measures how closely predicted bounding boxes generated by the model are aligned with the actual (ground truth) bounding boxes.
"""

# text_image = Text(
#     "<img src='https://github.com/dataset-ninja/model-benchmark-template/assets/78355358/8d7c63d0-2f3b-4f3f-9fd8-c6383a4bfba4' alt='alt text' width='300' />"
# )
# text_info = Text(
#     "To measure it, we calculate the <b>Intersection over Union (IoU)</b>. Intuitively, the higher the IoU, the closer two bounding boxes are. IoU is calculated by dividing the <b>area of overlap</b> between the predicted bounding box and the ground truth bounding box by the <b>area of union</b> of these two boxes.",
#     "info",
# )
markdown_iou_calculation = """<img src='https://github.com/dataset-ninja/model-benchmark-template/assets/78355358/8d7c63d0-2f3b-4f3f-9fd8-c6383a4bfba4' alt='alt text' width='300' />

To measure it, we calculate the <b>Intersection over Union (IoU)</b>. Intuitively, the higher the IoU, the closer two bounding boxes are. IoU is calculated by dividing the <b>area of overlap</b> between the predicted bounding box and the ground truth bounding box by the <b>area of union</b> of these two boxes.
"""

markdown_iou_distribution = """### IoU Distribution

This histogram represents the distribution of <abbr title="{}">IoU scores</abbr> between all predictions and their matched ground truth objects. This gives you a sense of how well the model aligns bounding boxes. Ideally, if the model aligns boxes very well, rightmost bars (from 0.9 to 1.0 IoU) should be much higher than others.
"""


notification_avg_iou = {
    "title": "Avg. IoU = {}",
    "description": "",
}

markdown_calibration_score_1 = """## Calibration Score

This section analyzes <abbr title="{}">confidence scores</abbr> (or predicted probabilities) that the model generates for every predicted bounding box.
"""

markdown_what_is_calibration = """In some applications, it's crucial for a model not only to make accurate predictions but also to provide reliable **confidence levels**. A well-calibrated model aligns its confidence scores with the actual likelihood of predictions being correct. For example, if a model claims 90% confidence for predictions but they are correct only half the time, it is **overconfident**. Conversely, **underconfidence** occurs when a model assigns lower confidence scores than the actual likelihood of its predictions. In the context of autonomous driving, this might cause a vehicle to brake or slow down too frequently, reducing travel efficiency and potentially causing traffic issues."""
markdown_calibration_score_2 = """To evaluate the calibration, we draw a <b>Reliability Diagram</b> and calculate <b>Expected Calibration Error</b> (ECE)."""

# text_info = Text(
#     "To evaluate the calibration, we draw a <b>Reliability Diagram</b> and calculate <b>Expected Calibration Error</b> (ECE) and <b>Maximum Calibration Error</b> (MCE).",
#     "info",
# )
markdown_reliability_diagram = """### Reliability Diagram

Reliability diagram, also known as a Calibration curve, helps in understanding whether the confidence scores of detections accurately represent the true probability of a correct detection. A well-calibrated model means that when it predicts a detection with, say, 80% confidence, approximately 80% of those predictions should actually be correct.
"""

markdown_calibration_curve_interpretation = """
1. **The curve is above the Ideal Line (Underconfidence):** If the calibration curve is consistently above the ideal line, this indicates underconfidence. The model's predictions are more correct than the confidence scores suggest. For example, if the model predicts a detection with 70% confidence but, empirically, 90% of such detections are correct, the model is underconfident.
2. **The curve is below the Ideal Line (Overconfidence):** If the calibration curve is below the ideal line, the model exhibits overconfidence. This means it is too sure of its predictions. For instance, if the model predicts with 80% confidence but only 60% of these predictions are correct, it is overconfident.

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

This chart breaks down the Confidence Profile into multiple curves, each for one <abbr title="{}">IoU threshold</abbr>. In this way you can understand how the f1-optimal confidence threshold changes with various IoU thresholds. Higher IoU thresholds mean that the model should align bounding boxes very close to ground truth bounding boxes.
"""
markdown_confidence_distribution = """### Confidence Distribution

This graph helps to assess whether high confidence scores correlate with correct detections (<abbr title="{}">True Positives</abbr>) and low confidence scores are mostly associated with incorrect detections (<abbr title="{}">False Positives</abbr>).

Additionally, it provides a view of how predicted probabilities are distributed. Whether the model skews probabilities to lower or higher values, leading to imbalance?

Ideally, the histogram for TP predictions should have higher confidence, indicating that the model is sure about its correct predictions, and the FP predictions should have very low confidence, or not present at all.
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

markdown_normalization = "By default, the normalization is used for better intraclass comparison. The total outcome counts are divided by the number of ground truth instances of the corresponding class. This is useful, because the sum of TP + FN always gives 1.0, representing all ground truth instances for a class, that gives a visual understanding of what portion of instances the model detected. So, if a green bar (TP outcomes) reaches the 1.0, this means the model is managed to predict all objects for the class. Everything that is higher than 1.0 corresponds to False Positives, i.e, redundant predictions. You can turn off the normalization switching to absolute values."

markdown_class_outcome_counts_2 = (
    """You can switch the plot view between normalized and absolute values.

_Bars in the chart are sorted by <abbr title="{}">F1-score</abbr> to keep a unified order of classes between different charts._

"""
    + clickable_label
)

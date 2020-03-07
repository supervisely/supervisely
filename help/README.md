## Tutorials and examples of working with Supervisely SDK

The documentation is still under active development.
For now we have the following resources available:

### How it works

* [Exam report explanation](./tutorials/06_exam_report_explanation/06_exam_report_explanation.md)

### Integration guides

* [Create a new Supervisely plugin](./tutorials/01_create_new_plugin/how_to_create_plugin.md)
* [Easy guide: Integrate a custom Pytorch Segmentation neural network](./tutorials/02_pytorch_easy_segmentation_plugin/pytorch_segmentation_integration_template.md)
* [General detailed guide: Integrate any custom neural network](./tutorials/03_custom_neural_net_plugin/custom_nn_plugin.md)
* [Deploy neural network as API](./tutorials/04_deploy_neural_net_as_api/deploy-model.md)
* [Develop NN plugin](./tutorials/05_develop_nn_plugin/develop_plugin.md)

### Python SDK tutorials and cookbooks


#### Tutorials

* [Working with Supervisely projects and labeling data](./jupyterlab_scripts/src/tutorials/01_project_structure/project.ipynb)
* [Scripting interactions with the web instance using Supervisely API](./jupyterlab_scripts/src/tutorials/02_data_management/data_management.ipynb)
* [Data augmentation for neural network training](./jupyterlab_scripts/src/tutorials/03_augmentations/augmentations.ipynb)
* [Neral network inference and online deployment using Supervisely API](./jupyterlab_scripts/src/tutorials/04_neural_network_inference/neural_network_inference.ipynb)
* [Automating comple workflows using Supervisely API](./jupyterlab_scripts/src/tutorials/05_neural_network_workflow/neural_network_workflow.ipynb)
* [Explanation of different inference modes for NN: full image, sliding window, roi](./jupyterlab_scripts/src/tutorials/06_inference_modes/inference_modes.ipynb)
* [How to copy, move and delete data using py-SDK and REST-API](./jupyterlab_scripts/src/tutorials/07_copy_move_delete_example/copy_move_delete.ipynb)
* [How to manage users and labeling jobs](./jupyterlab_scripts/src/tutorials/08_users_labeling_jobs_example/users_labeling_jobs_example.ipynb)
* [Custom inference pipeline: image -> detection -> segmentation -> postprocessing](./jupyterlab_scripts/src/tutorials/09_detection_segmentation_pipeline/detection_segmentation_pipeline.ipynb)
* [Custom upload procedure: check if image exists and how to upload only new images to Supervisely instance](./jupyterlab_scripts/src/tutorials/10_upload_only_new_images/upload_only_new_images.ipynb)
* [Custom data pipeline: upload -> auto labeling jobs -> move labeled data to "secret" project](./jupyterlab_scripts/src/tutorials/11_custom_data_pipeline/custom_data_pipeline.ipynb)
* [Filter images in different projects and combine results into a new project](./jupyterlab_scripts/src/tutorials/12_filter_and_combine_images/filter_combine_images.ipynb)
* [How to apply NN (UNet/YOLO/Mask-RCNN) to the image from sources](./jupyterlab_scripts/src/tutorials/13_nn_inference_from_sources/README.md)

#### Cookbooks

* [Analyse data annotation quality](./jupyterlab_scripts/src/cookbook/analyse_annotation_quality/analyse_annotation_quality.ipynb)
* [Calculate classification metrics](./jupyterlab_scripts/src/cookbook/calculate_classification_metrics/calculate_classification_metrics.ipynb)
* [Calculate confusion matrix](./jupyterlab_scripts/src/cookbook/calculate_confusion_matrix_metric/calculate_confusion_matrix.ipynb)
* [Calculate mean average precision (mAP)](./jupyterlab_scripts/src/cookbook/calculate_map_metric/calculate_map.ipynb)
* [Calculate mean intersection over union (mIOU)](./jupyterlab_scripts/src/cookbook/calculate_metrics/calculate_metrics.ipynb)
* [Convert between class geometry types](./jupyterlab_scripts/src/cookbook/convert_class_shape/convert_class_shape.ipynb)
* [Import a project using a list of image links](./jupyterlab_scripts/src/cookbook/create_project_from_links/create_project_from_links.ipynb)
* [Download a project locally](./jupyterlab_scripts/src/cookbook/download_project/download_project.ipynb)
* [Filter project by tags](./jupyterlab_scripts/src/cookbook/filter_project_by_tags/filter_project_by_tags.ipynb)
* [Merge projects into one](./jupyterlab_scripts/src/cookbook/merge_projects/merge_projects.ipynb)
* [Plot tags distribution statistics](./jupyterlab_scripts/src/cookbook/plot_tags_distribution/plot_tags_distribution.ipynb)
* [Split the data between train and validation folds using tags](./jupyterlab_scripts/src/cookbook/train_validation_tagging/train_validation_tagging.ipynb)
* [Add augmentations and prepare data for training a detection model](./jupyterlab_scripts/src/cookbook/training_data_for_detection/training_data_for_detection.ipynb)
* [Add augmentations and prepare data for training a segmentation model](./jupyterlab_scripts/src/cookbook/training_data_for_segmentation/training_data_for_segmentation.ipynb)
* [Upload a project using  using Supervisely API](./jupyterlab_scripts/src/cookbook/upload_project/upload_project.ipynb)


FROM docker.deepsystems.io/supervisely/five/base-py:latest

RUN pip install sphinx
RUN pip install sphinx-autoapi
RUN pip install recommonmark
RUN pip install sphinx-rtd-theme

RUN pip install pascal-voc-writer
RUN pip install nbsphinx
RUN conda install -c conda-forge pandoc

ENV PYTHONPATH /workdir:/workdir/src:/workdir/supervisely_lib/worker_proto:$PYTHONPATH

COPY . /sly

RUN mkdir /sly/docs/symlink
WORKDIR /sly/docs/symlink
COPY ./help/jupyterlab_scripts/src/tutorials/01_project_structure/project.ipynb \
       help/jupyterlab_scripts/src/tutorials/01_project_structure/project.ipynb
COPY ./help/jupyterlab_scripts/src/tutorials/02_data_management/data_management.ipynb \
       help/jupyterlab_scripts/src/tutorials/02_data_management/data_management.ipynb
COPY ./help/jupyterlab_scripts/src/tutorials/03_augmentations/augmentations.ipynb \
       help/jupyterlab_scripts/src/tutorials/03_augmentations/augmentations.ipynb
COPY ./help/jupyterlab_scripts/src/tutorials/04_neural_network_inference/neural_network_inference.ipynb \
       help/jupyterlab_scripts/src/tutorials/04_neural_network_inference/neural_network_inference.ipynb
COPY ./help/jupyterlab_scripts/src/tutorials/05_neural_network_workflow/neural_network_workflow.ipynb \
       help/jupyterlab_scripts/src/tutorials/05_neural_network_workflow/neural_network_workflow.ipynb
COPY ./help/jupyterlab_scripts/src/tutorials/06_inference_modes/inference_modes.ipynb \
       help/jupyterlab_scripts/src/tutorials/06_inference_modes/inference_modes.ipynb
COPY ./help/jupyterlab_scripts/src/tutorials/07_copy_move_delete_example/copy_move_delete.ipynb \
       help/jupyterlab_scripts/src/tutorials/07_copy_move_delete_example/copy_move_delete.ipynb
COPY ./help/jupyterlab_scripts/src/tutorials/08_users_labeling_jobs_example/users_labeling_jobs_example.ipynb \
       help/jupyterlab_scripts/src/tutorials/08_users_labeling_jobs_example/users_labeling_jobs_example.ipynb
COPY ./help/jupyterlab_scripts/src/tutorials/09_detection_segmentation_pipeline/detection_segmentation_pipeline.ipynb \
       help/jupyterlab_scripts/src/tutorials/09_detection_segmentation_pipeline/detection_segmentation_pipeline.ipynb
COPY ./help/jupyterlab_scripts/src/tutorials/10_upload_only_new_images/upload_only_new_images.ipynb \
       help/jupyterlab_scripts/src/tutorials/10_upload_only_new_images/upload_only_new_images.ipynb
COPY ./help/jupyterlab_scripts/src/tutorials/11_custom_data_pipeline/custom_data_pipeline.ipynb \
       help/jupyterlab_scripts/src/tutorials/11_custom_data_pipeline/custom_data_pipeline.ipynb
COPY ./help/jupyterlab_scripts/src/tutorials/12_filter_and_combine_images/filter_combine_images.ipynb \
       help/jupyterlab_scripts/src/tutorials/12_filter_and_combine_images/filter_combine_images.ipynb
#COPY ./help/jupyterlab_scripts/src/cookbook/upload_project/upload_project.ipynb \
#       help/jupyterlab_scripts/src/cookbook/upload_project/upload_project.ipynb
#COPY ./help/jupyterlab_scripts/src/cookbook/upload_images_via_api/upload_images_via_api.ipynb \
#       help/jupyterlab_scripts/src/cookbook/upload_images_via_api/upload_images_via_api.ipynb

COPY ./help/jupyterlab_scripts/src/tutorials/14_pixelwise_confidences/pixelwise_confidences.ipynb \
       help/jupyterlab_scripts/src/tutorials/14_pixelwise_confidences/pixelwise_confidences.ipynb

COPY ./README.md README.md


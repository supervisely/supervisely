create_jupyter_project () {
	local dir=$1
	local index=$2
  	item_name=$(echo ${dir##*/})
  	echo ${item_name}

  	project_path=${notebooks_dir}/${index}_${item_name}/workspace
  	#echo ${project_path}
  	mkdir -p ${project_path}
  	cp -r ${dir}/* ${project_path}

  	ipynb_checkpoints_path=${project_path}/.ipynb_checkpoints
  	#echo ${ipynb_checkpoints_path}
  	if [ -d {$ipynb_checkpoints_path} ]; then rm -rf {$ipynb_checkpoints_path}; fi

  	src_meta_path=${project_path}/meta.json
  	dst_meta_path=${notebooks_dir}/${index}_${item_name}/meta.json
  	#echo ${src_meta_path}
  	#echo ${dst_meta_path}
  	cp ${src_meta_path} ${dst_meta_path}
  	rm ${src_meta_path}  
}

# tutorials workspace
notebooks_dir=${PWD}/notebooks/tutorials
echo ${notebooks_dir}
if [ -d ${notebooks_dir} ]; then rm -rf ${notebooks_dir}; fi

create_jupyter_project "src/tutorials/01_project_structure" "001"
create_jupyter_project "src/tutorials/02_data_management" "002"
create_jupyter_project "src/tutorials/03_augmentations" "003"
create_jupyter_project "src/tutorials/04_neural_network_inference" "004"
create_jupyter_project "src/tutorials/05_neural_network_workflow" "005"
create_jupyter_project "src/tutorials/06_inference_modes" "006"
create_jupyter_project "src/tutorials/07_copy_move_delete_example" "007"
create_jupyter_project "src/tutorials/08_users_labeling_jobs_example" "008"
create_jupyter_project "src/tutorials/09_detection_segmentation_pipeline" "009"
create_jupyter_project "src/tutorials/10_upload_only_new_images" "010"
create_jupyter_project "src/tutorials/11_custom_data_pipeline" "011"
create_jupyter_project "src/tutorials/12_filter_and_combine_images" "012"


# cookbook workspace
notebooks_dir=${PWD}/notebooks/cookbook
echo ${notebooks_dir}
if [ -d ${notebooks_dir} ]; then rm -rf ${notebooks_dir}; fi

create_jupyter_project "src/cookbook/analyse_annotation_quality" "001"
create_jupyter_project "src/cookbook/calculate_classification_metrics" "002"
create_jupyter_project "src/cookbook/calculate_confusion_matrix_metric" "003"
create_jupyter_project "src/cookbook/calculate_map_metric" "004"
create_jupyter_project "src/cookbook/calculate_metrics" "005"
create_jupyter_project "src/cookbook/calculate_precision_recall_metric" "006"
create_jupyter_project "src/cookbook/convert_class_shape" "007"
create_jupyter_project "src/cookbook/create_project_from_links" "008"
create_jupyter_project "src/cookbook/download_project" "009"
create_jupyter_project "src/cookbook/filter_project_by_tags" "011"
create_jupyter_project "src/cookbook/merge_projects" "012"
create_jupyter_project "src/cookbook/plot_tags_distribution" "013"
create_jupyter_project "src/cookbook/tag_objects" "016"
create_jupyter_project "src/cookbook/train_validation_tagging" "017"
create_jupyter_project "src/cookbook/training_data_for_detection" "018"
create_jupyter_project "src/cookbook/training_data_for_segmentation" "019"
create_jupyter_project "src/cookbook/upload_images_via_api" "020"
create_jupyter_project "src/cookbook/upload_project" "021"








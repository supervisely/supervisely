from supervisely.nn.benchmark.comparison.base_comparison import BaseComparison


class ObjectDetectionComparison(BaseComparison):
    pass


# TODO: check if model evaluations of the same project or datasets (if datasets are provided)
# ? TODO: handle the case when the models evaluated on different classes
# ? TODO: handle the case when the models evaluated on different datasets


# usage example:
eval_dir_1 = "/model_benchmark/project_1/54601_serve_yolov8/evaluation/"
eval_dir_2 = "/model_benchmark/project_1/54602_serve_yolov8/evaluation/"
res_remote_dir = "/model_benchmark/project_1/comparison_results/"

bm_comparison = ObjectDetectionComparison(eval_dir_1, eval_dir_2) # ? , local=True) 
bm_comparison.run_compare()
bm_comparison.upload_results(res_remote_dir)
bm_comparison.get_report_link(res_remote_dir)

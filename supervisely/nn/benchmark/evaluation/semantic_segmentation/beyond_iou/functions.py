import json
import os
import random
import shutil
from os.path import join as pjoin
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage
from tqdm import tqdm

from supervisely.nn.benchmark.evaluation.semantic_segmentation.beyond_iou.evaluator import (
    Evaluator,
)
from supervisely.nn.benchmark.evaluation.semantic_segmentation.beyond_iou.loader import (
    build_segmentation_loader,
)
# from supervisely.nn.benchmark.evaluation.semantic_segmentation.image_diversity.clip_metrics import (
#     ClipMetrics,
# )
from supervisely.project.project import Project
from supervisely.sly_logger import logger

# # function for downloading project from supervisely platform to local storage
# def download_projects(api, gt_project_id, pred_project_id, dest_dir):
#     project_ids = [gt_project_id, pred_project_id]
#     project_types = ["ground truth", "predicted"]
#     for project_id, project_type in zip(project_ids, project_types):
#         if project_type == "ground truth":
#             project_dest_dir = f"{dest_dir}/gt/{gt_project_id}"
#         else:
#             project_dest_dir = f"{dest_dir}/pred/{pred_project_id}"
#         if os.path.exists(project_dest_dir):
#             sly.logger.info(f"Project already exists in {project_dest_dir} directory")
#         else:
#             project_info = api.project.get_info_by_id(project_id)
#             n_images = project_info.items_count
#             pbar = tqdm(desc=f"Downloading {project_type} project...", total=n_images)
#             Project.download(
#                 api=api,
#                 project_id=project_id,
#                 dest_dir=project_dest_dir,
#                 progress_cb=pbar,
#             )
#             sly.logger.info(f"Successfully downloaded {project_type} project")


# function for data preprocessing
def prepare_segmentation_data(source_project_dir, output_project_dir, palette):
    if os.path.exists(output_project_dir):
        logger.info(f"Preprocessed data already exists in {output_project_dir} directory")
        return
    else:
        os.makedirs(output_project_dir)

        ann_dir = "seg"

        temp_project_seg_dir = source_project_dir + "_temp"
        if not os.path.exists(temp_project_seg_dir):
            Project.to_segmentation_task(
                source_project_dir,
                temp_project_seg_dir,
            )

        datasets = os.listdir(temp_project_seg_dir)
        for dataset in datasets:
            if not os.path.isdir(os.path.join(temp_project_seg_dir, dataset)):
                continue
            # convert masks to required format and save to general ann_dir
            mask_files = os.listdir(os.path.join(temp_project_seg_dir, dataset, ann_dir))
            for mask_file in tqdm(mask_files, desc="Preparing segmentation data..."):
                mask = cv2.imread(os.path.join(temp_project_seg_dir, dataset, ann_dir, mask_file))[
                    :, :, ::-1
                ]
                result = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
                # human masks to machine masks
                for color_idx, color in enumerate(palette):
                    colormap = np.where(np.all(mask == color, axis=-1))
                    result[colormap] = color_idx
                if mask_file.count(".png") > 1:
                    mask_file = mask_file[:-4]
                cv2.imwrite(os.path.join(output_project_dir, mask_file), result)

        shutil.rmtree(temp_project_seg_dir)


# function for calculating performance metrics
def calculate_metrics(
    gt_dir,
    pred_dir,
    boundary_width,
    boundary_iou_d,
    num_workers,
    class_names,
    result_dir,
):
    if boundary_width % 1 == 0:
        boundary_width = int(boundary_width)
    evaluator = Evaluator(
        class_names=class_names,
        boundary_width=boundary_width,
        boundary_implementation="exact",
        boundary_iou_d=boundary_iou_d,
        result_dir=result_dir,
    )
    loader = build_segmentation_loader(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        gt_label_map=None,
        pred_label_map=None,
        num_workers=num_workers,
    )
    eval_data = evaluator.evaluate(loader)
    return eval_data
    # result.make_table(path=f"{result_dir}/table.md")


# # function for getting image ids for each cell of confusion matrix
# def get_cell_image_ids(api, project_id, cell_img_names):
#     name2id = {}
#     dataset_infos = api.dataset.get_list(project_id)
#     dataset_ids = [ds_info.id for ds_info in dataset_infos]
#     for dataset_id in dataset_ids:
#         img_infos = api.image.get_list(dataset_id)
#         for img_info in img_infos:
#             name2id[img_info.name] = img_info.id
#     cell_img_ids = {}
#     for cell, img_names in cell_img_names.items():
#         img_ids = [name2id[name] for name in img_names]
#         cell_img_ids[cell] = img_ids
#     return cell_img_ids


# # function for getting a set of random image names from specific project
# def get_random_image_paths(api, subset_size, project_id, except_list):
#     dataset_infos = api.dataset.get_list(project_id)
#     dataset_names = [ds.name for ds in dataset_infos]
#     image_paths = []
#     for dataset_name in dataset_names:
#         img_dir = f"sly_data/gt/{project_id}/{dataset_name}/img"
#         img_filenames = os.listdir(img_dir)
#         paths = [os.path.join(img_dir, filename) for filename in img_filenames]
#         image_paths.extend(paths)
#     image_paths = [img for img in image_paths if img not in except_list]
#     total_length = len(image_paths)
#     selected_indexes = [random.randint(0, total_length - 1) for i in range(subset_size)]
#     selected_image_paths = [image_paths[idx] for idx in selected_indexes]
#     return selected_image_paths


# # function for finding the most diverse subset of images from specific directory
# def get_diverse_images(api, subset_size, project_id, n_iter, batch_size, except_list):
#     diverse_image_names = None
#     highest_tce = float("-inf")
#     clip_metrics = ClipMetrics(n_eigs=subset_size - 1)

#     for iter in tqdm(
#         range(n_iter), desc="Creating diverse set of images for preview..."
#     ):
#         img_paths = get_random_image_paths(api, subset_size, project_id, except_list)
#         tce = clip_metrics.tce(img_paths=img_paths, batch_size=batch_size)
#         if tce > highest_tce:
#             highest_tce = tce
#             diverse_image_names = img_paths

#     diverse_image_names = [os.path.basename(path) for path in diverse_image_names]
#     return diverse_image_names


# # function for drawing preview images
# def draw_preview_set(
#     api,
#     gt_project_dir,
#     pred_project_dir,
#     gt_project_id,
#     preview_img_names,
#     output_dir,
# ):
#     if os.path.exists(output_dir):
#         shutil.rmtree(output_dir)
#     os.makedirs(output_dir)

#     img_dir, ann_dir = "img", "ann"

#     project_meta = sly.ProjectMeta.from_json(api.project.get_meta(gt_project_id))
#     dir_elements = os.listdir(gt_project_dir)

#     for element in dir_elements:
#         if not os.path.isdir(os.path.join(gt_project_dir, element)):
#             continue
#         img_files = os.listdir(os.path.join(gt_project_dir, element, img_dir))
#         img_files = [file for file in img_files if file in preview_img_names]

#         for img_file in img_files:
#             base_name, extension = img_file.split(".")
#             original_path = os.path.join(gt_project_dir, element, img_dir, img_file)
#             original_image = sly.image.read(original_path)

#             gt_ann_path = os.path.join(
#                 gt_project_dir, element, ann_dir, img_file + ".json"
#             )
#             with open(gt_ann_path, "r") as file:
#                 gt_ann_json = json.load(file)

#             gt_ann = sly.Annotation.from_json(gt_ann_json, project_meta)
#             gt_ann_image = np.copy(original_image)
#             gt_ann.draw_pretty(gt_ann_image, thickness=3)

#             pred_ann_path = os.path.join(
#                 pred_project_dir, element, ann_dir, img_file + ".json"
#             )
#             with open(pred_ann_path, "r") as file:
#                 pred_ann_json = json.load(file)

#             pred_ann = sly.Annotation.from_json(pred_ann_json, project_meta)
#             pred_ann_image = np.copy(original_image)
#             pred_ann.draw_pretty(pred_ann_image, thickness=3)

#             sly.image.write(
#                 f"{output_dir}/{base_name}_collage.{extension}",
#                 np.hstack((original_image, gt_ann_image, pred_ann_image)),
#             )


# function for drawing charts
# def draw_charts(df, class_names, result_dir="output"):
#     import plotly.graph_objects as go
#     from plotly.subplots import make_subplots
#     import plotly.figure_factory as ff

#     chart_dir = f"{result_dir}/charts"
#     if os.path.exists(chart_dir):
#         shutil.rmtree(chart_dir)
#     os.makedirs(chart_dir)
#     fig = make_subplots(
#         rows=1,
#         cols=3,
#         subplot_titles=(
#             "Basic segmentation metrics",
#             "Intersection & Error over Union",
#             "Renormalized Error over Union",
#         ),
#         specs=[[{"type": "polar"}, {"type": "domain"}, {"type": "xy"}]],
#     )
#     # first subplot
#     categories = [
#         "mPixel accuracy",
#         "mPrecision",
#         "mRecall",
#         "mF1-score",
#         "mIoU",
#         "mBoundaryIoU",
#         "mPixel accuracy",
#     ]
#     num_classes = len(df.index) - 1
#     precision = round(df.loc["mean", "precision"] * 100, 1)
#     recall = round(df.loc["mean", "recall"] * 100, 1)
#     f1_score = round(df.loc["mean", "F1_score"] * 100, 1)
#     iou = round(df.loc["mean", "IoU"] * 100, 1)
#     boundary_iou = round(df.loc["mean", "boundary_IoU"] * 100, 1)
#     overall_TP = df["TP"][:num_classes].sum()
#     overall_FN = df["FN"][:num_classes].sum()
#     pixel_accuracy = round((overall_TP / (overall_TP + overall_FN)) * 100, 1)
#     values = [
#         pixel_accuracy,
#         precision,
#         recall,
#         f1_score,
#         iou,
#         boundary_iou,
#         pixel_accuracy,
#     ]
#     trace_1 = go.Scatterpolar(
#         mode="lines+text",
#         r=values,
#         theta=categories,
#         fill="toself",
#         fillcolor="cornflowerblue",
#         line_color="blue",
#         opacity=0.6,
#         text=values,
#         textposition=[
#             "bottom right",
#             "top center",
#             "top center",
#             "middle left",
#             "bottom center",
#             "bottom right",
#             "bottom right",
#         ],
#         textfont=dict(color="blue"),
#     )
#     fig.add_trace(trace_1, row=1, col=1)
#     # second subplot
#     labels = ["mIoU", "mBoundaryEoU", "mExtentEoU", "mSegmentEoU"]
#     boundary_eou = round(df.loc["mean", "E_boundary_oU"] * 100, 1)
#     extent_eou = round(df.loc["mean", "E_extent_oU"] * 100, 1)
#     segment_eou = round(df.loc["mean", "E_segment_oU"] * 100, 1)
#     values = [iou, boundary_eou, extent_eou, segment_eou]
#     trace_2 = go.Pie(
#         labels=labels,
#         values=values,
#         hole=0.5,
#         textposition="outside",
#         textinfo="percent+label",
#         marker=dict(colors=["cornflowerblue", "moccasin", "lightgreen", "orangered"]),
#     )
#     fig.add_trace(trace_2, row=1, col=2)
#     # third subplot
#     labels = ["boundary", "extent", "segment"]
#     boundary_renormed_eou = round(df.loc["mean", "E_boundary_oU_renormed"] * 100, 1)
#     extent_renormed_eou = round(df.loc["mean", "E_extent_oU_renormed"] * 100, 1)
#     segment_renormed_eou = round(df.loc["mean", "E_segment_oU_renormed"] * 100, 1)
#     values = [boundary_renormed_eou, extent_renormed_eou, segment_renormed_eou]
#     trace_3 = go.Bar(
#         x=labels,
#         y=values,
#         orientation="v",
#         text=values,
#         width=[0.5, 0.5, 0.5],
#         textposition="outside",
#         marker_color=["moccasin", "lightgreen", "orangered"],
#     )
#     fig.add_trace(trace_3, row=1, col=3)
#     fig.update_layout(
#         height=400,
#         width=1200,
#         polar=dict(
#             radialaxis=dict(
#                 visible=True, showline=False, showticklabels=False, range=[0, 100]
#             )
#         ),
#         showlegend=False,
#         plot_bgcolor="rgba(0, 0, 0, 0)",
#         yaxis=dict(showticklabels=False),
#         yaxis_range=[0, int(max(values)) + 4],
#     )
#     fig.layout.annotations[0].update(y=1.2)
#     fig.layout.annotations[1].update(y=1.2)
#     fig.layout.annotations[2].update(y=1.2)
#     fig.write_image(f"{chart_dir}/overall_error_analysis.png", scale=2.0)
#     fig.write_html(f"{chart_dir}/overall_error_analysis.html")
#     overall_chart_img = sly.image.read(f"{chart_dir}/overall_error_analysis.png")
#     chart_height, chart_width = overall_chart_img.shape[0], overall_chart_img.shape[1]
#     # fourth plot
#     df.drop(["mean"], inplace=True)
#     if len(df.index) > 7:
#         per_class_iou = df["IoU"].copy()
#         per_class_iou.sort_values(ascending=True, inplace=True)
#         target_classes = per_class_iou.index[:7].tolist()
#         title_text = "Classwise segmentation error analysis<br><sup>(7 classes with highest error rates)</sup>"
#         labels = target_classes[::-1]
#         bar_data = df.loc[target_classes].copy()
#     else:
#         title_text = "Classwise segmentation error analysis"
#         bar_data = df.copy()
#     bar_data = bar_data[["IoU", "E_extent_oU", "E_boundary_oU", "E_segment_oU"]]
#     bar_data.sort_values(by="IoU", ascending=False, inplace=True)
#     if not len(df.index) > 7:
#         labels = list(bar_data.index)
#     color_palette = ["cornflowerblue", "moccasin", "lightgreen", "orangered"]

#     fig = go.Figure()
#     for i, column in enumerate(bar_data.columns):
#         fig.add_trace(
#             go.Bar(
#                 name=column,
#                 y=bar_data[column],
#                 x=labels,
#                 marker_color=color_palette[i],
#             )
#         )
#     fig.update_yaxes(range=[0, 1])
#     fig.update_layout(
#         barmode="stack",
#         plot_bgcolor="rgba(0, 0, 0, 0)",
#         title={
#             "text": title_text,
#             "y": 0.96,
#             "x": 0.5,
#             "xanchor": "center",
#             "yanchor": "top",
#         },
#         font=dict(size=24),
#     )
#     fig.write_image(
#         f"{chart_dir}/classwise_error_analysis.png",
#         scale=1.0,
#         width=chart_width,
#         height=chart_height,
#     )
#     fig.write_html(f"{chart_dir}/classwise_error_analysis.html")
#     # fifth plot
#     confusion_matrix = np.load(f"{result_dir}/confusion_matrix.npy")
#     if len(df.index) > 7:
#         original_classes = df.index.tolist()
#         remove_classes = per_class_iou.index[7:].tolist()
#         remove_indexes = [original_classes.index(cls) for cls in remove_classes]
#         # for index in remove_indexes:
#         #     confusion_matrix = np.delete(confusion_matrix, index, 0)
#         #     confusion_matrix = np.delete(confusion_matrix, index, 1)
#         #     class_names.remove(original_classes[index])
#         confusion_matrix = np.delete(confusion_matrix, remove_indexes, 0)
#         confusion_matrix = np.delete(confusion_matrix, remove_indexes, 1)
#         class_names = [cls for cls in class_names if cls not in remove_classes]
#         title_text = (
#             "Confusion matrix<br><sup>(7 classes with highest error rates)</sup>"
#         )
#     else:
#         title_text = "Confusion matrix"

#     confusion_matrix = confusion_matrix[::-1]
#     x = class_names
#     y = x[::-1].copy()
#     text_anns = [[str(el) for el in row] for row in confusion_matrix]

#     fig = ff.create_annotated_heatmap(
#         confusion_matrix, x=x, y=y, annotation_text=text_anns, colorscale="orrd"
#     )

#     fig.update_layout(
#         title={
#             "text": title_text,
#             "y": 0.97,
#             "x": 0.5,
#             "xanchor": "center",
#             "yanchor": "top",
#         },
#         font=dict(size=24),
#     )

#     fig.add_annotation(
#         dict(
#             font=dict(color="black", size=24),
#             x=0.5,
#             y=-0.1,
#             showarrow=False,
#             text="Predicted",
#             xref="paper",
#             yref="paper",
#         )
#     )
#     fig.add_annotation(
#         dict(
#             font=dict(color="black", size=24),
#             x=-0.12,
#             y=0.5,
#             showarrow=False,
#             text="Ground truth",
#             textangle=-90,
#             xref="paper",
#             yref="paper",
#         )
#     )

#     fig.update_layout(margin=dict(t=150, l=300))
#     fig["data"][0]["showscale"] = True
#     fig.write_image(
#         f"{chart_dir}/confusion_matrix.png",
#         scale=1.0,
#         width=chart_width,
#         height=chart_height,
#     )
#     fig.write_html(f"{chart_dir}/confusion_matrix.html")
#     # sixth plot
#     confusion_matrix = np.load(f"{result_dir}/confusion_matrix.npy")
#     n_pairs = 10

#     non_diagonal_indexes = {}
#     for i, idx in enumerate(np.ndindex(confusion_matrix.shape)):
#         if idx[0] != idx[1]:
#             non_diagonal_indexes[i] = idx

#     indexes_1d = np.argsort(confusion_matrix, axis=None)
#     indexes_2d = [
#         non_diagonal_indexes[idx] for idx in indexes_1d if idx in non_diagonal_indexes
#     ][-n_pairs:]
#     indexes_2d = np.asarray(indexes_2d[::-1])

#     rows = indexes_2d[:, 0]
#     cols = indexes_2d[:, 1]
#     probs = confusion_matrix[rows, cols]

#     confused_classes = []
#     for idx in indexes_2d:
#         gt_idx, pred_idx = idx[0], idx[1]
#         gt_class = class_names[gt_idx]
#         pred_class = class_names[pred_idx]
#         confused_classes.append(f"{gt_class}-{pred_class}")

#     fig = go.Figure()
#     fig.add_trace(
#         go.Bar(
#             x=confused_classes,
#             y=probs,
#             orientation="v",
#             text=probs,
#         )
#     )
#     fig.update_traces(
#         textposition="outside",
#         marker=dict(color=probs, colorscale="orrd"),
#     )
#     fig.update_layout(
#         plot_bgcolor="rgba(0, 0, 0, 0)",
#         yaxis_range=[0, max(probs) + 0.1],
#         yaxis=dict(showticklabels=False),
#     )
#     fig.update_layout(
#         title={
#             "text": "Frequently confused classes",
#             "y": 0.9,
#             "x": 0.5,
#             "xanchor": "center",
#             "yanchor": "top",
#         },
#         font=dict(size=24),
#     )
#     fig.write_image(
#         f"{chart_dir}/frequently_confused.png",
#         scale=1.0,
#         width=chart_width,
#         height=chart_height,
#     )
#     fig.write_html(f"{chart_dir}/frequently_confused.html")
#     # combine all chart images into one
#     classwise_chart_img = sly.image.read(f"{chart_dir}/classwise_error_analysis.png")
#     conf_matrix_chart_img = sly.image.read(f"{chart_dir}/confusion_matrix.png")
#     freq_confused_chart_img = sly.image.read(f"{chart_dir}/frequently_confused.png")
#     sly.image.write(
#         f"{chart_dir}/total_dashboard.png",
#         np.vstack(
#             (
#                 overall_chart_img,
#                 classwise_chart_img,
#                 conf_matrix_chart_img,
#                 freq_confused_chart_img,
#             )
#         ),
#     )


# # unified function for evaluating segmentation quality
# def evaluate_segmentation_quality(
#     gt_project_path,
#     pred_project_path,
#     # subset_size=3,
#     # n_iter=15,
#     # batch_size=8,
#     result_dir="output",
# ):

#     meta_path = pjoin(gt_project_path, "meta.json")
#     project_meta = sly.ProjectMeta.from_json(load_json_file(meta_path))
#     obj_classes = project_meta.obj_classes
#     classes_json = obj_classes.to_json()
#     class_names = [obj["title"] for obj in classes_json]
#     palette = [obj["color"].lstrip("#") for obj in classes_json]
#     palette = [[int(color[i : i + 2], 16) for i in (0, 2, 4)] for color in palette]

#     gt_prep_path = Path(gt_project_path).parent / "preprocessed_gt"
#     pred_prep_path = Path(pred_project_path).parent / "preprocessed_pred"
#     prepare_segmentation_data(gt_project_path, gt_prep_path, palette)
#     prepare_segmentation_data(pred_project_path, pred_prep_path, palette)

#     calculate_metrics(
#         gt_dir=gt_prep_path,
#         pred_dir=pred_prep_path,
#         boundary_width=0.01,
#         boundary_iou_d=0.02,
#         num_workers=4,
#         output_dir=result_dir,
#         class_names=class_names,
#         result_dir=result_dir,
#     )
#     sly.logger.info("Successfully calculated evaluation metrics")

#     with open(f"{result_dir}/cell_img_names.json", "r") as file:
#         cell_img_names = json.load(file)

#     cell_img_ids = get_cell_image_ids(api, pred_project_id, cell_img_names)

#     with open(f"{result_dir}/cell_img_ids.json", "w") as file:
#         json.dump(cell_img_ids, file)

#     metrics_df = pd.read_csv(f"{result_dir}/per_image_metrics.csv", index_col=0)
#     per_image_ious = metrics_df["iou"].sort_values(ascending=False)
#     lowest_iou_images = list(per_image_ious[-subset_size:].index)
#     highest_iou_images = list(per_image_ious[:subset_size].index)

#     except_list = lowest_iou_images + highest_iou_images
#     diverse_images = get_diverse_images(
#         api, subset_size, gt_project_id, n_iter, batch_size, except_list
#     )
#     preview_img_names = highest_iou_images + lowest_iou_images + diverse_images

#     draw_preview_set(
#         api=api,
#         gt_project_dir=gt_sly_dir,
#         pred_project_dir=pred_sly_dir,
#         gt_project_id=gt_project_id,
#         preview_img_names=preview_img_names,
#         output_dir=f"{result_dir}/preview_set",
#     )
#     sly.logger.info(
#         f"Successfully created preview set of images in {result_dir}/preview_set directory"
#     )

#     df = pd.read_csv(f"{result_dir}/result_df.csv", index_col="Unnamed: 0")
#     draw_charts(df, class_names, result_dir)
#     sly.logger.info(
#         f"Successfully saved performance charts in {result_dir}/charts directory"
#     )

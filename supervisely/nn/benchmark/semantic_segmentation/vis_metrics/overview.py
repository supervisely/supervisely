import datetime
from typing import List

from supervisely.nn.benchmark.semantic_segmentation.base_vis_metric import (
    SemanticSegmVisMetric,
)
from supervisely.nn.benchmark.visualization.widgets import MarkdownWidget


class Overview(SemanticSegmVisMetric):

    def get_header(self, user_login: str) -> MarkdownWidget:
        current_date = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
        header_text = self.vis_texts.markdown_header.format(
            self.eval_result.name, user_login, current_date
        )
        header = MarkdownWidget("markdown_header", "Header", text=header_text)
        return header

    @property
    def overview_md(self) -> List[MarkdownWidget]:
        url = self.eval_result.inference_info.get("checkpoint_url")
        link_text = self.eval_result.inference_info.get("custom_checkpoint_path")
        if link_text is None:
            link_text = url
        link_text = link_text.replace("_", "\_")

        model_name = self.eval_result.inference_info.get("model_name") or "Custom"
        checkpoint_name = self.eval_result.checkpoint_name

        # Note about validation dataset
        classes_str, note_about_images, starter_app_info = self._get_overview_info()

        formats = [
            model_name.replace("_", "\_"),
            checkpoint_name.replace("_", "\_"),
            self.eval_result.inference_info.get("architecture"),
            self.eval_result.inference_info.get("task_type"),
            self.eval_result.inference_info.get("runtime"),
            url,
            link_text,
            self.eval_result.gt_project_info.id,
            self.eval_result.gt_project_info.name,
            classes_str,
            note_about_images,
            starter_app_info,
            self.vis_texts.docs_url,
        ]

        md = MarkdownWidget(
            "markdown_overview",
            "Overview",
            text=self.vis_texts.markdown_overview.format(*formats),
        )
        md.is_info_block = True
        md.width_fit_content = True
        return md

    def _get_overview_info(self):
        classes_cnt = len(self.eval_result.classes_whitelist)
        classes_str = "classes" if classes_cnt > 1 else "class"
        classes_str = f"{classes_cnt} {classes_str}"

        evaluator_session, train_session, images_str = None, None, ""
        gt_project_id = self.eval_result.gt_project_info.id
        gt_dataset_ids = self.eval_result.gt_dataset_ids
        gt_images_cnt = self.eval_result.val_images_cnt
        train_info = self.eval_result.train_info
        evaluator_app_info = self.eval_result.evaluator_app_info
        total_imgs_cnt = self.eval_result.gt_project_info.items_count
        if gt_images_cnt is not None:
            val_imgs_cnt = gt_images_cnt
        elif gt_dataset_ids is not None:
            datasets = self.eval_result.gt_dataset_infos
            val_imgs_cnt = sum(ds.items_count for ds in datasets)
        else:
            val_imgs_cnt = total_imgs_cnt

        if train_info:
            train_task_id = train_info.get("app_session_id")
            if train_task_id:
                app_id = self.eval_result.task_info["meta"]["app"]["id"]
                train_session = f'- **Training dashboard**:  <a href="/apps/{app_id}/sessions/{train_task_id}" target="_blank">open</a>'

            train_imgs_cnt = train_info.get("images_count")
            images_str = f", {train_imgs_cnt} images in train, {val_imgs_cnt} images in validation"

        if gt_images_cnt is not None:
            images_str += (
                f", total {total_imgs_cnt} images. Evaluated using subset - {val_imgs_cnt} images"
            )
        elif gt_dataset_ids is not None:
            links = [
                f'<a href="/projects/{gt_project_id}/datasets/{ds.id}" target="_blank">{ds.name}</a>'
                for ds in datasets
            ]
            images_str += f", total {total_imgs_cnt} images. Evaluated on the dataset{'s' if len(links) > 1 else ''}: {', '.join(links)}"
        else:
            images_str += f", total {total_imgs_cnt} images. Evaluated on the whole project ({val_imgs_cnt} images)"

        if evaluator_app_info:
            evaluator_task_id = evaluator_app_info.get("id")
            evaluator_app_id = evaluator_app_info.get("meta", {}).get("app", {}).get("id")
            evaluator_app_name = evaluator_app_info.get("meta", {}).get("app", {}).get("name")
            if evaluator_task_id and evaluator_app_id and evaluator_app_name:
                evaluator_session = f'- **Evaluator app session**:  <a href="/apps/{evaluator_app_id}/sessions/{evaluator_task_id}" target="_blank">open</a>'

        starter_app_info = train_session or evaluator_session or ""

        return classes_str, images_str, starter_app_info

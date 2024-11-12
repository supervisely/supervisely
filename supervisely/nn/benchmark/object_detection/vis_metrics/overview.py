import datetime
from typing import List

from supervisely.nn.benchmark.object_detection.base_vis_metric import DetectionVisMetric
from supervisely.nn.benchmark.visualization.widgets import MarkdownWidget


class Overview(DetectionVisMetric):

    def get_header(self, user_login: str) -> MarkdownWidget:
        current_date = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
        header_text = self.vis_texts.markdown_header.format(
            self.eval_result.name, user_login, current_date
        )
        header = MarkdownWidget("markdown_header", "Header", text=header_text)
        return header

    @property
    def md(self) -> List[MarkdownWidget]:
        url = self.eval_result.inference_info.get("checkpoint_url")
        link_text = self.eval_result.inference_info.get("custom_checkpoint_path")
        if link_text is None:
            link_text = url
        link_text = link_text.replace("_", "\_")

        model_name = self.eval_result.inference_info.get("model_name") or "Custom"
        checkpoint_name = self.eval_result.inference_info.get("deploy_params", {}).get(
            "checkpoint_name", ""
        )

        # Note about validation dataset
        classes_str, note_about_val_dataset, train_session = self._get_overview_info()

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
            note_about_val_dataset,
            train_session,
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

        train_session, images_str = "", ""
        gt_project_id = self.eval_result.gt_project_info.id
        gt_dataset_ids = self.eval_result.gt_dataset_ids
        gt_images_ids = self.eval_result.gt_images_ids
        train_info = self.eval_result.train_info
        if gt_images_ids is not None:
            val_imgs_cnt = len(gt_images_ids)
        elif gt_dataset_ids is not None:
            datasets = self.eval_result.gt_dataset_infos
            val_imgs_cnt = sum(ds.items_count for ds in datasets)
        else:
            val_imgs_cnt = self.eval_result.gt_project_info.items_count

        if train_info:
            train_task_id = train_info.get("app_session_id")
            if train_task_id:
                task_info = self.eval_result.train_info
                app_id = task_info["meta"]["app"]["id"]
                train_session = f'- **Training dashboard**:  <a href="/apps/{app_id}/sessions/{train_task_id}" target="_blank">open</a>'

            train_imgs_cnt = train_info.get("images_count")
            images_str = f", {train_imgs_cnt} images in train, {val_imgs_cnt} images in validation"

        if gt_images_ids is not None:
            images_str += f". Evaluated using subset - {val_imgs_cnt} images"
        elif gt_dataset_ids is not None:
            links = [
                f'<a href="/projects/{gt_project_id}/datasets/{ds.id}" target="_blank">{ds.name}</a>'
                for ds in datasets
            ]
            images_str += (
                f". Evaluated on the dataset{'s' if len(links) > 1 else ''}: {', '.join(links)}"
            )
        else:
            images_str += f". Evaluated on the whole project ({val_imgs_cnt} images)"

        return classes_str, images_str, train_session

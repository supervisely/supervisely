import datetime
from typing import List

from supervisely.nn.benchmark.object_detection.base_vis_metric import DetectionVisMetric
from supervisely.nn.benchmark.visualization.widgets import MarkdownWidget, TableWidget


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
            link_text = url or ""
        link_text = link_text.replace("_", "\_")

        model_name = self.eval_result.inference_info.get("model_name") or "Custom"
        checkpoint_name = self.eval_result.checkpoint_name

        # Note about validation dataset
        classes_str, note_about_images, starter_app_info = self._get_overview_info()

        # link to scroll to the optimal confidence section
        opt_conf_url = self.vis_texts.docs_url + "#f1-optimal-confidence-threshold"
        average_url = self.vis_texts.docs_url + "#averaging-iou-thresholds"

        iou_threshold = self.eval_result.mp.iou_threshold
        if self.eval_result.different_iou_thresholds_per_class:
            iou_threshold = "Different IoU thresholds for each class (see the table below)"

        conf_text = (
            f"- **Optimal confidence threshold**: "
            f"{round(self.eval_result.mp.f1_optimal_conf, 4)} (calculated automatically), "
            f"<a href='{opt_conf_url}' target='_blank'>learn more</a>."
        )
        custom_conf_thrs = self.eval_result.mp.custom_conf_threshold
        if custom_conf_thrs is not None:
            conf_text += f"\n- **Custom confidence threshold**: {custom_conf_thrs}"

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
            iou_threshold,
            conf_text,
            self.eval_result.mp.average_across_iou_thresholds,
            average_url,
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
            val_imgs_cnt = self.eval_result.pred_project_info.items_count

        if train_info:
            train_task_id = train_info.get("app_session_id")
            if train_task_id:
                app_id = self.eval_result.task_info["meta"]["app"]["id"]
                train_session = f'- **Training dashboard**:  <a href="/apps/{app_id}/sessions/{train_task_id}" target="_blank">open</a>'

            train_imgs_cnt = train_info.get("images_count")
            images_str = f", {train_imgs_cnt} images in train, {val_imgs_cnt} images in validation"

        if gt_images_cnt is not None:
            images_str += (
                f", total {total_imgs_cnt} images. Evaluated using subset - {gt_images_cnt} images"
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

    @property
    def iou_per_class_md(self) -> List[MarkdownWidget]:
        if not self.eval_result.different_iou_thresholds_per_class:
            return None

        return MarkdownWidget(
            "markdown_iou_per_class",
            "Different IoU thresholds for each class",
            text=self.vis_texts.markdown_iou_per_class,
        )

    @property
    def iou_per_class_table(self) -> TableWidget:
        if not self.eval_result.different_iou_thresholds_per_class:
            return None

        content = []
        for name, thr in self.eval_result.mp.iou_threshold_per_class.items():
            row = [name, round(thr, 2)]
            dct = {"row": row, "id": name, "items": row}
            content.append(dct)

        data = {
            "columns": ["Class name", "IoU threshold"],
            "columnsOptions": [{"disableSort": True}, {}],
            "content": content,
        }
        return TableWidget(
            name="table_iou_per_class",
            data=data,
            fix_columns=1,
            width="60%",
            main_column="Class name",
        )

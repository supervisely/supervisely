from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Set

from supervisely import Api, logger
from supervisely._utils import generate_free_name

DATASET_ITEMS = "items"
NESTED_DATASETS = "datasets"


@dataclass
class ProjectStructureUploader:
    """
    Uploads a nested project structure of datasets.

    The structure is expected to be a dict:
      {
        "ds_name": {
          "items": [<converter items>],
          "datasets": { ... nested ... }
        },
        ...
      }
    """

    existing_datasets: Set[str]

    def upload(
        self,
        api: Api,
        project_id: int,
        root_dataset_id: int,
        project_structure: Dict,
        upload_items: Callable[[int, Iterable[Any]], None],
    ) -> None:
        def _walk(
            node: Dict,
            parent_id: Optional[int],
            first_dataset: bool,
            reuse_dataset_id: Optional[int] = None,
        ) -> None:
            for ds_name, value in node.items():
                ds_name = generate_free_name(
                    self.existing_datasets, ds_name, extend_used_names=True
                )

                if first_dataset and reuse_dataset_id is not None:
                    dataset_id = reuse_dataset_id
                    api.dataset.update(dataset_id, ds_name)  # rename first dataset
                    first_dataset = False
                else:
                    dataset_id = api.dataset.create(project_id, ds_name, parent_id=parent_id).id

                items = value.get(DATASET_ITEMS, []) or []
                nested = value.get(NESTED_DATASETS, {}) or {}

                logger.info(
                    f"Dataset: {ds_name}, items: {len(items)}, nested datasets: {len(nested)}"
                )

                if items:
                    upload_items(dataset_id, items)

                if nested:
                    _walk(nested, parent_id=dataset_id, first_dataset=False)

        _walk(
            project_structure,
            parent_id=None,
            first_dataset=True,
            reuse_dataset_id=root_dataset_id,
        )

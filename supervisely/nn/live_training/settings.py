from dataclasses import dataclass, field
from typing import List, Optional

import supervisely as sly


class OverlapMode:
    """How overlapping masks are resolved when rasterizing a semantic segmentation mask."""

    LAST_ADDED = "last_added"  # object order in the annotation: a later object wins
    CLASS_PRIORITY = "class_priority"  # class_order[0] is drawn on top

    ALL = (LAST_ADDED, CLASS_PRIORITY)


@dataclass
class LiveTrainingSettings:
    """
    Session setup sent by the web in the body of ``POST /start``:

    .. code-block:: json

        {
          "classes": [101, 102],
          "ignore_unselected": true,
          "overlap": "class_priority",
          "class_order": [102, 101],
          "seed": {"dataset_ids": [11, 12, 13], "filters": [...]}
        }

    ``seed`` may be ``null``: the user skipped the "pretrain on existing annotations" step.

    The settings are fixed for the whole session and returned unchanged by ``/settings``.
    """

    classes: List[int] = field(default_factory=list)
    ignore_unselected: bool = False
    overlap: str = OverlapMode.LAST_ADDED
    class_order: List[int] = field(default_factory=list)
    seed_dataset_ids: List[int] = field(default_factory=list)
    seed_filters: List[dict] = field(default_factory=list)

    @property
    def has_seed(self) -> bool:
        return len(self.seed_dataset_ids) > 0

    @classmethod
    def from_json(cls, data: Optional[dict]) -> Optional["LiveTrainingSettings"]:
        """Parse settings from the /start payload. Returns None if the payload carries no settings."""
        if not data or not isinstance(data, dict):
            return None
        if data.get("classes") is None:
            # No class selection: old web or a launch without the setup dialog.
            return None
        seed = data.get("seed") or {}
        return cls(
            classes=[int(c) for c in data.get("classes") or []],
            ignore_unselected=bool(data.get("ignore_unselected", False)),
            overlap=data.get("overlap") or OverlapMode.LAST_ADDED,
            class_order=[int(c) for c in data.get("class_order") or []],
            seed_dataset_ids=[int(d) for d in seed.get("dataset_ids") or []],
            seed_filters=list(seed.get("filters") or []),
        )

    @classmethod
    def default(cls, eligible_classes: List[sly.ObjClass]) -> "LiveTrainingSettings":
        """Today's behaviour: train all eligible classes, later object wins, no seed."""
        class_ids = [obj_class.sly_id for obj_class in eligible_classes]
        return cls(classes=class_ids, class_order=list(class_ids))

    def to_json(self) -> dict:
        seed = None
        if self.has_seed:
            seed = {"dataset_ids": list(self.seed_dataset_ids), "filters": list(self.seed_filters)}
        return {
            "classes": list(self.classes),
            "ignore_unselected": self.ignore_unselected,
            "overlap": self.overlap,
            "class_order": list(self.class_order),
            "seed": seed,
        }

    def validate(self, project_meta: sly.ProjectMeta, eligible_classes: List[sly.ObjClass]):
        """
        Validate settings against the project and normalize ``class_order``.

        :raises ValueError: if settings are invalid.
        """
        if self.overlap not in OverlapMode.ALL:
            raise ValueError(
                f"Unknown overlap mode '{self.overlap}'. Expected one of: {list(OverlapMode.ALL)}"
            )
        if not self.classes:
            raise ValueError("No classes selected for training. Select at least one class.")

        self.classes = list(dict.fromkeys(self.classes))
        eligible_ids = {obj_class.sly_id for obj_class in eligible_classes}
        project_ids = {obj_class.sly_id: obj_class for obj_class in project_meta.obj_classes}
        for class_id in self.classes:
            if class_id not in project_ids:
                raise ValueError(f"Class with id {class_id} not found in the project.")
            if class_id not in eligible_ids:
                obj_class = project_ids[class_id]
                raise ValueError(
                    f"Class '{obj_class.name}' ({obj_class.geometry_type.geometry_name()}) "
                    f"can't be used for this task."
                )

        unknown_order = [c for c in self.class_order if c not in self.classes]
        if unknown_order:
            raise ValueError(f"class_order contains classes that are not selected: {unknown_order}")
        class_order = list(dict.fromkeys(self.class_order))
        class_order += [c for c in self.classes if c not in class_order]
        self.class_order = class_order

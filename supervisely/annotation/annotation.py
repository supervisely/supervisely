# coding: utf-8
"""annotation for a single image"""

# docs
from __future__ import annotations

import itertools
import json
import operator
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image

from supervisely import logger
from supervisely._utils import take_with_default
from supervisely.annotation.label import Label, LabelJsonFields
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_collection import TagCollection
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.image_rotator import ImageRotator
from supervisely.geometry.multichannel_bitmap import MultichannelBitmap
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging import font as sly_font
from supervisely.imaging import image as sly_image
from supervisely.io.fs import ensure_base_path
from supervisely.project.project_meta import ProjectMeta

if TYPE_CHECKING:
    try:
        from imgaug.augmentables.bbs import BoundingBoxesOnImage
        from imgaug.augmentables.segmaps import SegmentationMapsOnImage
    except ModuleNotFoundError:
        pass

ANN_EXT = ".json"


class AnnotationJsonFields:
    """
    Json fields for :class:`Annotation<supervisely.annotation.annotation.Annotation>`
    """

    IMG_DESCRIPTION = "description"
    """"""
    IMG_SIZE = "size"
    """"""
    IMG_SIZE_WIDTH = "width"
    """"""
    IMG_SIZE_HEIGHT = "height"
    """"""
    IMG_TAGS = "tags"
    """"""
    LABELS = "objects"
    """"""
    CUSTOM_DATA = "customBigData"
    """"""
    PROBABILITY_CLASSES = "probabilityClasses"
    """"""
    PROBABILITY_LABELS = "probabilityLabels"
    """"""
    IMAGE_ID = "imageId"
    """"""


class Annotation:
    """
    Annotation for a single image. :class:`Annotation<Annotation>` object is immutable.

    :param img_size: Size of the image (height, width).
    :type img_size: Tuple[int, int] or List[int, int]
    :param labels: List of Label objects.
    :type labels: List[Label]
    :param img_tags: TagCollection object or list of Tag objects.
    :type img_tags: TagCollection or List[Tag]
    :param img_description: Image description.
    :type img_description: str, optional
    :param pixelwise_scores_labels: List of Label objects.
    :type pixelwise_scores_labels: List[Label]
    :param custom_data: Custom data.
    :type custom_data: dict, optional
    :param image_id: Id of the image.
    :type image_id: int, optional

    :raises: :class:`TypeError`, if image size is not tuple or list
    :Usage example:

     .. code-block:: python

        # Simple Annotation example
        import supervisely as sly

        height, width = 500, 700
        ann = sly.Annotation((height, width))

        # More complex Annotation example
        # TagCollection
        meta_lemon = sly.TagMeta('lemon_tag', sly.TagValueType.ANY_STRING)
        tag_lemon = sly.Tag(meta_lemon, 'Hello')
        tags = sly.TagCollection([tag_lemon])
        # or tags = [tag_lemon]

        # ObjClass
        class_lemon = sly.ObjClass('lemon', sly.Rectangle)

        # Label
        label_lemon = sly.Label(sly.Rectangle(100, 100, 200, 200), class_lemon)

        # Annotation
        height, width = 300, 400
        ann = sly.Annotation((height, width), [label_lemon], tags, 'example annotaion')
        # 'points': {'exterior': [[100, 100], [200, 200]], 'interior': []}

        # If Label geometry is out of image size bounds, it will be cropped
        label_lemon = sly.Label(sly.Rectangle(100, 100, 700, 900), class_lemon)
        height, width = 300, 400

        ann = sly.Annotation((height, width), [label_lemon], tags, 'example annotaion')
        # 'points': {'exterior': [[100, 100], [399, 299]], 'interior': []}
    """

    def __init__(
        self,
        img_size: Union[Tuple[int, int], Tuple[None, None]],
        labels: Optional[List[Label]] = None,
        img_tags: Optional[Union[TagCollection, List[Tag]]] = None,
        img_description: Optional[str] = "",
        pixelwise_scores_labels: Optional[List[Label]] = None,
        custom_data: Optional[Dict] = None,
        image_id: Optional[int] = None,
    ):
        if not isinstance(img_size, (tuple, list)):
            raise TypeError(
                '{!r} has to be a tuple or a list. Given type "{}".'.format(
                    "img_size", type(img_size)
                )
            )
        self._img_size = tuple(img_size)
        if self._img_size.count(None) == 1:
            raise ValueError("Image resolution (height, width) has to defined both or none of them")

        self._img_description = img_description

        if img_tags is None:
            self._img_tags = TagCollection()
        elif isinstance(img_tags, list):
            self._img_tags = TagCollection(img_tags)
        elif isinstance(img_tags, TagCollection):
            self._img_tags = img_tags
        else:
            raise TypeError(f"img_tags argument has unknown type {type(img_tags)}")

        self._labels = []
        self._add_labels_impl(self._labels, take_with_default(labels, []))
        self._pixelwise_scores_labels = []  # @TODO: store pixelwise scores as usual geometry labels
        self._add_labels_impl(
            self._pixelwise_scores_labels,
            take_with_default(pixelwise_scores_labels, []),
        )
        self._custom_data = take_with_default(custom_data, {})
        self._image_id = image_id

    @property
    def img_size(self) -> Tuple[int, int]:
        """
        Size of the image (height, width).

        :return: Image size
        :rtype: :class:`Tuple[int, int]`
        :Usage example:

         .. code-block:: python

            height, width = 300, 400
            ann = sly.Annotation((height, width))
            print(ann.img_size)
            # Output: (300, 400)
        """
        return deepcopy(self._img_size)

    @property
    def image_id(self) -> int:
        """
        Id of the image.

        :return: Image id
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            height, width = 300, 400
            image_id = 12345
            ann = sly.Annotation((height, width), image_id=image_id)
            print(ann.image_id)
            # Output: 12345
        """
        return self._image_id

    @property
    def labels(self) -> List[Label]:
        """
        Labels on annotation.

        :return: Copy of list with image labels
        :rtype: :class:`List[Label]<supervisely.annotation.label.Label>`
        :Usage example:

         .. code-block:: python

            # Create Labels and add them to Annotation
            class_kiwi = sly.ObjClass('kiwi', sly.Rectangle)
            label_kiwi = sly.Label(sly.Rectangle(0, 0, 300, 300), class_kiwi)

            class_lemon = sly.ObjClass('lemon', sly.Rectangle)
            label_lemon = sly.Label(sly.Rectangle(0, 0, 500, 600), class_lemon)

            labels_arr = [label_kiwi, label_lemon]

            height, width = 300, 400
            ann = sly.Annotation((height, width), labels_arr)

            # Note that ann.labels return a COPY of list with image labels
            class_potato = sly.ObjClass('potato', sly.Rectangle)
            label_potato = sly.Label(sly.Rectangle(0, 0, 200, 400), class_potato)

            ann.labels.append(label_potato)
            print(len(ann.labels))
            # Output: 2

            ann_arr = ann.labels
            ann_arr.append(label_potato)
            print(len(ann_arr))
            # Output: 3
        """
        return self._labels.copy()

    @property
    def pixelwise_scores_labels(self):
        """pixelwise_scores_labels"""
        return self._pixelwise_scores_labels.copy()

    @property
    def img_description(self) -> str:
        """
        Image description.

        :return: Image description
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            ann = sly.Annotation((500, 700), img_description='Annotation for this image is empty')
            print(ann.img_description)
            # Output: Annotation for this image is empty
        """
        return self._img_description

    @property
    def img_tags(self) -> TagCollection:
        """
        Image tags.

        :return: TagCollection object
        :rtype: :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>`
        :Usage example:

         .. code-block:: python

            # Create TagCollection
            meta_weather = sly.TagMeta('weather', sly.TagValueType.ANY_STRING)
            tag_weather = sly.Tag(meta_weather, 'cloudy')
            tags = sly.TagCollection([tag_weather])

            ann = sly.Annotation((300, 400), img_tags=tags)
            print(ann.img_tags)
            # Output:
            #   Tags:
            #   +----------------+------------+--------+
            #   |      Name      | Value type | Value  |
            #   +----------------+------------+--------+
            #   |     weather    | any_string | cloudy |
            #   +----------------+------------+--------+
        """
        return self._img_tags

    def to_json(self) -> Dict:
        """
        Convert the Annotation to a json dict. Read more about `Supervisely format <https://docs.supervisely.com/data-organization/00_ann_format_navi>`_.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            ann = sly.Annotation((500, 700))
            ann_json = ann.to_json()

            print(ann_json)
            # Output: {
            #     "description": "",
            #     "size": {
            #         "height": 500,
            #         "width": 700
            #     },
            #     "tags": [],
            #     "objects": [],
            #     "customBigData": {}
            # }
        """
        height = self.img_size[0]
        if height is not None:
            height = int(height)

        width = self.img_size[1]
        if width is not None:
            width = int(width)
        res = {
            AnnotationJsonFields.IMG_DESCRIPTION: self.img_description,
            AnnotationJsonFields.IMG_SIZE: {
                AnnotationJsonFields.IMG_SIZE_HEIGHT: height,
                AnnotationJsonFields.IMG_SIZE_WIDTH: width,
            },
            AnnotationJsonFields.IMG_TAGS: self.img_tags.to_json(),
            AnnotationJsonFields.LABELS: [label.to_json() for label in self.labels],
            AnnotationJsonFields.CUSTOM_DATA: self.custom_data,
        }
        if len(self._pixelwise_scores_labels) > 0:
            # construct probability classes from labels
            prob_classes = {}
            for label in self._pixelwise_scores_labels:
                # @TODO: hotfix to save geometry as "multichannelBitmap" instead of "bitmap"; use normal classes
                prob_classes[label.obj_class.name] = label.obj_class.clone(
                    geometry_type=MultichannelBitmap
                )

            # save probabilities
            probabilities = {
                AnnotationJsonFields.PROBABILITY_LABELS: [
                    label.to_json() for label in self._pixelwise_scores_labels
                ],
                AnnotationJsonFields.PROBABILITY_CLASSES: ObjClassCollection(
                    list(prob_classes.values())
                ).to_json(),
            }
            res[AnnotationJsonFields.CUSTOM_DATA].update(probabilities)
        if self.image_id is not None:
            res[AnnotationJsonFields.IMAGE_ID] = self.image_id
        return res

    @classmethod
    def from_json(cls, data: Dict, project_meta: ProjectMeta) -> Annotation:
        """
        Convert a json dict to Annotation. Read more about `Supervisely format <https://docs.supervisely.com/data-organization/00_ann_format_navi>`_.

        :param data: Annotation in json format as a dict.
        :type data: dict
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`.
        :type project_meta: ProjectMeta
        :return: Annotation object
        :rtype: :class:`Annotation<Annotation>`
        :raises: :class:`Exception`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta = sly.ProjectMeta()

            ann_json = {
                 "size": {
                     "height": 500,
                     "width": 700
                 },
                 "tags": [],
                 "objects": []
            }

            ann = sly.Annotation.from_json(ann_json, meta)
        """
        img_size_dict = data[AnnotationJsonFields.IMG_SIZE]
        img_height = img_size_dict[AnnotationJsonFields.IMG_SIZE_HEIGHT]
        img_width = img_size_dict[AnnotationJsonFields.IMG_SIZE_WIDTH]
        img_size = (img_height, img_width)
        try:
            labels = [
                Label.from_json(label_json, project_meta)
                for label_json in data[AnnotationJsonFields.LABELS]
            ]
        except Exception as e:
            raise RuntimeError(
                f"Failed to deserialize one of the label from JSON format annotation: \n{repr(e)}"
            )

        custom_data = data.get(AnnotationJsonFields.CUSTOM_DATA, {}) or {}
        prob_labels = None
        if (
            AnnotationJsonFields.PROBABILITY_LABELS in custom_data
            and AnnotationJsonFields.PROBABILITY_CLASSES in custom_data
        ):
            prob_classes = ObjClassCollection.from_json(
                custom_data[AnnotationJsonFields.PROBABILITY_CLASSES]
            )

            # @TODO: tony, maybe link with project meta (add probability classes???)
            prob_project_meta = ProjectMeta(obj_classes=prob_classes)
            prob_labels = [
                Label.from_json(label_json, prob_project_meta)
                for label_json in custom_data[AnnotationJsonFields.PROBABILITY_LABELS]
            ]

            custom_data.pop(AnnotationJsonFields.PROBABILITY_CLASSES)
            custom_data.pop(AnnotationJsonFields.PROBABILITY_LABELS)

        image_id = data.get(AnnotationJsonFields.IMAGE_ID, None)

        return cls(
            img_size=img_size,
            labels=labels,
            img_tags=TagCollection.from_json(
                data[AnnotationJsonFields.IMG_TAGS], project_meta.tag_metas
            ),
            img_description=data.get(AnnotationJsonFields.IMG_DESCRIPTION, ""),
            pixelwise_scores_labels=prob_labels,
            custom_data=custom_data,
            image_id=image_id,
        )

    @classmethod
    def load_json_file(cls, path: str, project_meta: ProjectMeta) -> Annotation:
        """
        Loads json file and converts it to Annotation.

        :param path: Path to the json file.
        :type path: str
        :param project_meta: Input ProjectMeta object.
        :type project_meta: ProjectMeta
        :return: Annotation object
        :rtype: :class:`Annotation<Annotation>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            team_name = 'Vehicle Detection'
            workspace_name = 'Cities'
            project_name =  'London'

            team = api.team.get_info_by_name(team_name)
            workspace = api.workspace.get_info_by_name(team.id, workspace_name)
            project = api.project.get_info_by_name(workspace.id, project_name)

            meta = api.project.get_meta(project.id)

            # Load json file
            path = "/home/admin/work/docs/my_dataset/ann/annotation.json"
            ann = sly.Annotation.load_json_file(path, meta)
        """
        with open(path) as fin:
            data = json.load(fin)
        return cls.from_json(data, project_meta)

    def clone(
        self,
        img_size: Optional[Tuple[int, int]] = None,
        labels: Optional[List[Label]] = None,
        img_tags: Optional[Union[TagCollection, List[Tag]]] = None,
        img_description: Optional[str] = None,
        pixelwise_scores_labels: Optional[List[Label]] = None,
        custom_data: Optional[Dict] = None,
        image_id: Optional[int] = None,
    ) -> Annotation:
        """
        Makes a copy of Annotation with new fields, if fields are given, otherwise it will use fields of the original Annotation.

        :param img_size: Size of the image (height, width).
        :type img_size: Tuple[int, int] or List[int, int]
        :param labels: List of Label objects.
        :type labels: List[Label]
        :param img_tags: TagCollection object or list of Tag objects.
        :type img_tags: TagCollection or List[Tag]
        :param img_description: Image description.
        :type img_description: str, optional
        :param pixelwise_scores_labels: List of Label objects.
        :type pixelwise_scores_labels: List[Label]
        :param custom_data: Custom data.
        :type custom_data: dict, optional
        :param image_id: Id of the image.
        :type image_id: int, optional

        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`
        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            ann = sly.Annotation((300, 400))

            # Let's clone our Annotation with Label
            class_kiwi = sly.ObjClass('kiwi', sly.Rectangle)
            label_kiwi = sly.Label(sly.Rectangle(0, 0, 300, 300), class_kiwi)

            # Assign cloned annotation to a new variable
            ann_clone_1 = ann.clone(labels=[label_kiwi])

            # Let's clone our Annotation with Label, TagCollection and description
            meta_lemon = sly.TagMeta('lemon', sly.TagValueType.ANY_STRING)
            tag_lemon = sly.Tag(meta_lemon, 'juicy')
            tags = sly.TagCollection([tag_lemon])

            # Assign cloned annotation to a new variable
            ann_clone_2 = ann.clone(labels=[label_kiwi], img_tags=tags, img_description='Juicy')

        """
        return Annotation(
            img_size=take_with_default(img_size, self.img_size),
            labels=take_with_default(labels, self.labels),
            img_tags=take_with_default(img_tags, self.img_tags),
            img_description=take_with_default(img_description, self.img_description),
            pixelwise_scores_labels=take_with_default(
                pixelwise_scores_labels, self.pixelwise_scores_labels
            ),
            custom_data=take_with_default(custom_data, self.custom_data),
            image_id=take_with_default(image_id, self.image_id),
        )

    def _add_labels_impl(self, dest: List, labels: List[Label]):
        """
        The function _add_labels_impl extend list of the labels of the current Annotation object
        :param dest: destination list of the Label class objects
        :param labels: list of the Label class objects to be added to the destination list
        :return: list of the Label class objects
        """
        for label in labels:
            if self.img_size.count(None) == 0:
                # image has resolution in DB
                canvas_rect = Rectangle.from_size(self.img_size)
                try:
                    dest.extend(label.crop(canvas_rect))
                except Exception:
                    logger.error(
                        f"Cannot crop label of '{label.obj_class.name}' class "
                        "when extend list of the labels of the current Annotation object",
                        exc_info=True,
                    )
                    raise
            else:
                # image was uploaded by link and does not have resolution in DB
                # add label without normalization and validation
                dest.append(label)

    def add_label(self, label: Label) -> Annotation:
        """
        Clones Annotation and adds a new Label.

        :param label: Label to be added.
        :type label: Label
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`
        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            ann = sly.Annotation((300, 600))

            # Create Label
            class_kiwi = sly.ObjClass('kiwi', sly.Rectangle)
            label_kiwi = sly.Label(sly.Rectangle(0, 0, 300, 300), class_kiwi)

            # Add label to Annotation
            # Remember that Annotation object is immutable, and we need to assign new instance of Annotation to a new variable
            new_ann = ann.add_label(label_kiwi)
        """
        return self.add_labels([label])

    def add_labels(self, labels: List[Label]) -> Annotation:
        """
        Clones Annotation and adds a new Labels.

        :param labels: List of Label objects to be added.
        :type labels: List[Label]
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`
        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            ann = sly.Annotation((300, 600))

            # Create Labels
            class_kiwi = sly.ObjClass('kiwi', sly.Rectangle)
            label_kiwi = sly.Label(sly.Rectangle(0, 0, 300, 300), class_kiwi)

            class_lemon = sly.ObjClass('lemon', sly.Rectangle)
            label_lemon = sly.Label(sly.Rectangle(0, 0, 500, 600), class_lemon)

            # Add labels to Annotation
            # Remember that Annotation object is immutable, and we need to assign new instance of Annotation to a new variable
            new_ann = ann.add_labels([label_kiwi, label_lemon])
        """
        new_labels = []
        self._add_labels_impl(new_labels, labels)
        return self.clone(labels=[*self._labels, *new_labels])

    def delete_label(self, label: Label) -> Annotation:
        """
        Clones Annotation with removed Label.

        :param label: Label to be deleted.
        :type label: Label
        :raises: :class:`KeyError`, if there is no deleted Label in current Annotation object
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`
        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            ann = sly.Annotation((300, 600))

            # Create Labels
            class_kiwi = sly.ObjClass('kiwi', sly.Rectangle)
            label_kiwi = sly.Label(sly.Rectangle(0, 0, 300, 300), class_kiwi)

            class_lemon = sly.ObjClass('lemon', sly.Rectangle)
            label_lemon = sly.Label(sly.Rectangle(0, 0, 500, 600), class_lemon)

            # Add labels to Annotation
            ann = ann.add_labels([label_kiwi, label_lemon])
            print(len(ann.labels))
            # Output: 2

            # Run through all labels in Annotation objects
            for label in ann.labels:
                if label.obj_class.name == 'lemon': # label obj_class name we want to delete
                    # Remember that Annotation object is immutable, and we need to assign new instance of Annotation to a new variable
                    new_ann = ann.delete_label(label)

            print(len(new_ann.labels))
            # Output: 1
        """
        retained_labels = [_label for _label in self._labels if _label != label]
        if len(retained_labels) == len(self._labels):
            raise KeyError(
                "Trying to delete a non-existing label of class: {}".format(label.obj_class.name)
            )
        return self.clone(labels=retained_labels)

    def add_pixelwise_score_label(self, label: Label) -> Annotation:
        """
        Add label to the pixelwise_scores_labels and return the copy of the current  Annotation object.
        :param label: Label class object to be added
        :return: Annotation class object with the new list of the pixelwise_scores_labels
        """
        return self.add_pixelwise_score_labels([label])

    def add_pixelwise_score_labels(self, labels: List[Label]) -> Annotation:
        """
        Add_pixelwise_score_labels extend list of the labels of the pixelwise_scores_labels and return
        the copy of the current  Annotation object.
        :param labels: list of the Label class objects to be added
        :return: Annotation class object with the new list of the pixelwise_scores_labels
        """
        new_labels = []
        self._add_labels_impl(new_labels, labels)
        return self.clone(pixelwise_scores_labels=[*self._pixelwise_scores_labels, *new_labels])

    def add_tag(self, tag: Tag) -> Annotation:
        """
        Clones Annotation and adds a new Tag.

        :param tag: Tag object to be added.
        :type tag: Tag
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            ann = sly.Annotation((300, 600))

            # Create Tag
            meta_message = sly.TagMeta('Message', sly.TagValueType.ANY_STRING)
            tag_message = sly.Tag(meta_message, 'Hello')

            # Add Tag to Annotation
            # Remember that Annotation object is immutable, and we need to assign new instance of Annotation to a new variable
            new_ann = ann.add_tag(tag_message)
        """
        return self.clone(img_tags=self._img_tags.add(tag))

    def add_tags(self, tags: List[Tag]) -> Annotation:
        """
        Clones Annotation and adds a new list of Tags.

        :param tags: List of Tags to be added.
        :type tags: List[Tag]
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            ann = sly.Annotation((300, 600))

            # Create Tags
            meta_message = sly.TagMeta('Message', sly.TagValueType.ANY_STRING)
            meta_alert = sly.TagMeta('Alert', sly.TagValueType.NONE)

            tag_message = sly.Tag(meta_message, 'Hello')
            tag_alert = sly.Tag(meta_alert)

            # Add Tags to Annotation
            # Remember that Annotation object is immutable, and we need to assign new instance of Annotation to a new variable
            new_ann = ann.add_tags([tag_message, tag_alert])
        """
        return self.clone(img_tags=self._img_tags.add_items(tags))

    def delete_tags_by_name(self, tag_names: List[str]) -> Annotation:
        """
        Clones Annotation and removes Tags by their names.

        :param tag_names: List of Tags names to be deleted.
        :type tag_names: List[str]
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            ann = sly.Annotation((300, 600))

            # Create Tags
            meta_message = sly.TagMeta('Message', sly.TagValueType.ANY_STRING)
            meta_alert = sly.TagMeta('Alert', sly.TagValueType.NONE)

            tag_message = sly.Tag(meta_message, 'Hello')
            tag_alert = sly.Tag(meta_alert)

            # Add Tags to Annotation
            tags_ann = ann.add_tags([tag_message, tag_alert])

            # Delete Tags from Annotation
            # Remember that Annotation object is immutable, and we need to assign new instance of Annotation to a new variable
            new_ann = tags_ann.delete_tags_by_name(['Message', 'Alert'])
        """
        retained_tags = [tag for tag in self._img_tags.items() if tag.meta.name not in tag_names]
        return self.clone(img_tags=TagCollection(items=retained_tags))

    def delete_tag_by_name(self, tag_name: str) -> Annotation:
        """
        Clones Annotation with removed Tag by it's name.

        :param tag_name: Tag name to be delete.
        :type tag_name: str
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            ann = sly.Annotation((300, 600))

            # Create Tag
            meta_alert = sly.TagMeta('Alert', sly.TagValueType.ANY_STRING)
            tag_alert = sly.Tag(meta_alert, 'Hello')

            tag_ann = ann.add_tag(tag_alert)

            # Delete Tag from Annotation
            # Remember that Annotation object is immutable, and we need to assign new instance of Annotation to a new variable
            new_ann = tag_ann.delete_tag_by_name('Alert')
        """
        return self.delete_tags_by_name([tag_name])

    def delete_tags(self, tags: List[Tag]) -> Annotation:
        """
        Clones Annotation with removed Tags.

        :param tags: List of Tags to be deleted.
        :type tags: List[Tag]
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            ann = sly.Annotation((300, 600))

            meta_message = sly.TagMeta('Message', sly.TagValueType.ANY_STRING)
            meta_alert = sly.TagMeta('Alert', sly.TagValueType.NONE)

            tag_message = sly.Tag(meta_message, 'Hello')
            tag_alert = sly.Tag(meta_alert)

            ann = ann.add_tags([tag_message, tag_alert])
            print(len(ann.img_tags))
            # Output: 2

            new_ann = ann.delete_tags([tag_message, tag_alert])
            print(len(new_ann.img_tags))
            # Output: 0
        """
        return self.delete_tags_by_name([tag.meta.name for tag in tags])

    def delete_tag(self, tag: Tag) -> Annotation:
        """
        Clones Annotation with removed Tag.

        :param tag: Tag to be deleted.
        :type tag: Tag
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            ann = sly.Annotation((300, 600))

            meta_dog = sly.TagMeta('dog', sly.TagValueType.ANY_STRING)
            meta_cat = sly.TagMeta('cat', sly.TagValueType.NONE)

            tag_dog = sly.Tag(meta_dog, 'Woof!')
            tag_cat = sly.Tag(meta_cat)

            ann = ann.add_tags([tag_dog, tag_cat])
            print(len(ann.img_tags))
            # Output: 2

            new_ann = ann.delete_tag(tag_dog)
            print(len(new_ann.img_tags))
            # Output: 1
        """
        return self.delete_tags_by_name([tag.meta.name])

    def transform_labels(
        self, label_transform_fn, new_size: Optional[Tuple[int, int]] = None
    ) -> Annotation:
        """
        Transform labels and change image size in current Annotation object and return the copy of the current
        Annotation object.
        :param label_transform_fn: function for transform labels
        :param new_size: new image size
        :return: Annotation class object with new labels and image size
        """

        def _do_transform_labels(src_labels, label_transform_fn):
            # long easy to debug
            # result = []
            # for label in src_labels:
            #     result.extend(label_transform_fn(label))
            # return result

            # short, hard-to-debug alternative
            return list(itertools.chain(*[label_transform_fn(label) for label in src_labels]))

        new_labels = _do_transform_labels(self._labels, label_transform_fn)
        new_pixelwise_scores_labels = _do_transform_labels(
            self._pixelwise_scores_labels, label_transform_fn
        )
        return self.clone(
            img_size=take_with_default(new_size, self.img_size),
            labels=new_labels,
            pixelwise_scores_labels=new_pixelwise_scores_labels,
        )

    def crop_labels(self, rect: Rectangle) -> Annotation:
        """
        Crops Labels of the current Annotation.

        :param rect: Rectangle object for crop.
        :type rect: Rectangle
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image and annotation from API
            project_id = 888
            image_id = 555555

            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_info = api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            img = api.image.download_np(image_id)
            new_img = copy.deepcopy(img)

            # Draw Annotation on image before crop
            ann.draw_pretty(img, thickness=3)

            # Crop Labels for current Annotation
            # Remember that Annotation object is immutable, and we need to assign new instance of Annotation to a new variable
            cropped_ann = ann.crop_labels(sly.Rectangle(0, 0, 600, 700))

            # Draw Annotation on image after crop
            cropped_ann.draw_pretty(new_img, thickness=3)

        .. list-table::

            * - .. figure:: https://i.imgur.com/6huO1se.jpg

                   Before

              - .. figure:: https://i.imgur.com/w2wR4h8.jpg

                   After
        """

        def _crop_label(label):
            return label.crop(rect)

        return self.transform_labels(_crop_label)

    def relative_crop(self, rect: Rectangle) -> Annotation:
        """
        Crops current Annotation and with image size (height, width) changes.

        :param rect: Rectangle object for crop.
        :type rect: Rectangle
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image and annotation from API
            project_id = 888
            image_id = 555555

            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_info = api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            img = api.image.download_np(image_id)
            new_img = copy.deepcopy(img)
            new_img = sly.imaging.image.crop(new_img, sly.Rectangle(200, 300, 600, 700))

            # Draw Annotation on image before relative crop
            ann.draw_pretty(img, thickness=3)

            # Relative Crop Labels for current Annotation
            # Remember that Annotation object is immutable, and we need to assign new instance of Annotation to a new variable
            r_cropped_ann = ann.relative_crop(sly.Rectangle(200, 300, 600, 700))

            # Draw Annotation on image after relative crop
            r_cropped_ann.draw_pretty(new_img, thickness=3)

        .. list-table::

            * - .. figure:: https://i.imgur.com/23UuNdJ.png

                   Before

              - .. figure:: https://i.imgur.com/8Z7xVxB.jpg

                   After
        """

        def _crop_label(label):
            return label.relative_crop(rect)

        return self.transform_labels(_crop_label, rect.to_size())

    def rotate(self, rotator: ImageRotator) -> Annotation:
        """
        Rotates current Annotation.

        :param rotator: ImageRotator object.
        :type rotator: ImageRotator
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.image_rotator import ImageRotator

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image and annotation from API
            project_id = 888
            image_id = 555555

            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_info = api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            img = api.image.download_np(image_id)
            new_img = copy.deepcopy(img)
            new_img = sly.imaging.image.rotate(new_img, 10)

            # Draw Annotation on image before rotation
            ann.draw_pretty(img, thickness=3)

            # Rotate Labels for current Annotation
            # Remember that Annotation object is immutable, and we need to assign new instance of Annotation to a new variable
            rotator = ImageRotator(annotation.img_size, 10)
            rotated_ann = ann.rotate(rotator)

            # Draw Annotation on image after rotation
            rotated_ann.draw_pretty(new_img, thickness=3)

        .. list-table::

            * - .. figure:: https://i.imgur.com/6huO1se.jpg

                   Before

              - .. figure:: https://i.imgur.com/ZQ47cXN.jpg

                   After
        """

        def _rotate_label(label):
            return [label.rotate(rotator)]

        return self.transform_labels(_rotate_label, tuple(rotator.new_imsize))

    def resize(self, out_size: Tuple[int, int], skip_empty_masks: bool = False) -> Annotation:
        """
        Resizes current Annotation.

        :param out_size: Desired output image size (height, width).
        :type out_size: Tuple[int, int]
        :param skip_empty_masks: Skip the raising of the error when you have got an empty label mask after a resizing procedure.
        :type skip_empty_masks: bool

        :return: New instance of Annotation
        :rtype: :class: Annotation

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image and annotation from API
            project_id = 888
            image_id = 555555

            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_info = api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            img = api.image.download_np(image_id)
            new_img = copy.deepcopy(img)
            new_img = sly.imaging.image.resize(new_img, (100, 200))

            # Draw Annotation on image before resize
            ann.draw_pretty(img, thickness=3)

            # Resize
            # Remember that Annotation object is immutable, and we need to assign new instance of Annotation to a new variable
            resized_ann = ann.resize((100, 200))

            # Draw Annotation on image after resize
            resized_ann.draw_pretty(new_img, thickness=3)

        .. list-table::

            * - .. figure:: https://i.imgur.com/6huO1se.jpg

                   Before

              - .. figure:: https://i.imgur.com/RrvNMoV.jpg

                   After
        """

        def _resize_label(label):
            try:
                return [label.resize(self.img_size, out_size)]
            except ValueError:
                if skip_empty_masks is True:
                    return []
                else:
                    raise

        return self.transform_labels(_resize_label, out_size)

    def scale(self, factor: float) -> Annotation:
        """
        Scales current Annotation with the given factor.

        :param factor: Scale size.
        :type factor: float
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image and annotation from API
            project_id = 888
            image_id = 555555

            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_info = api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            img = api.image.download_np(image_id)
            new_img = copy.deepcopy(img)
            new_img = sly.imaging.image.scale(new_img, 0.55)

            # Draw Annotation on image before rescale
            ann.draw_pretty(img, thickness=3)

            # Scale
            # Remember that Annotation object is immutable, and we need to assign new instance of Annotation to a new variable
            rescaled_ann = ann.scale(0.55)

            # Draw Annotation on image after rescale
            rescaled_ann.draw_pretty(new_img, thickness=3)

        .. list-table::

            * - .. figure:: https://i.imgur.com/6huO1se.jpg

                   Before

              - .. figure:: https://i.imgur.com/Ze6uqZ8.jpg

                   After
        """

        def _scale_label(label):
            return [label.scale(factor)]

        result_size = (
            round(self.img_size[0] * factor),
            round(self.img_size[1] * factor),
        )
        return self.transform_labels(_scale_label, result_size)

    def fliplr(self) -> Annotation:
        """
        Flips the current Annotation horizontally.

        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image and annotation from API
            project_id = 888
            image_id = 555555

            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_info = api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            img = api.image.download_np(image_id)
            new_img = copy.deepcopy(img)
            new_img = sly.imaging.image.fliplr(new_img)

            # Draw Annotation on image before horizontal flip
            ann.draw_pretty(img, thickness=3)

            # Flip
            # Remember that Annotation object is immutable, and we need to assign new instance of Annotation to a new variable
            fliplr_ann = ann.fliplr()

            # Draw Annotation on image after horizontal flip
            fliplr_ann.draw_pretty(new_img, thickness=3)

        .. list-table::

            * - .. figure:: https://i.imgur.com/6huO1se.jpg

                   Before

              - .. figure:: https://i.imgur.com/AQSuqIN.jpg

                   After
        """

        def _fliplr_label(label):
            return [label.fliplr(self.img_size)]

        return self.transform_labels(_fliplr_label)

    def flipud(self) -> Annotation:
        """
        Flips the current Annotation vertically.

        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image and annotation from API
            project_id = 888
            image_id = 555555

            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_info = api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            img = api.image.download_np(image_id)
            new_img = copy.deepcopy(img)
            new_img = sly.imaging.image.flipud(new_img)

            # Draw Annotation on image before vertical flip
            ann.draw_pretty(img, thickness=3)

            # Flip
            # Remember that Annotation object is immutable, and we need to assign new instance of Annotation to a new variable
            flipud_ann = ann.flipud()

            # Draw Annotation on image after vertical flip
            flipud_ann.draw_pretty(new_img, thickness=3)

        .. list-table::

            * - .. figure:: https://i.imgur.com/6huO1se.jpg

                   Before

              - .. figure:: https://i.imgur.com/NVhvPDb.jpg

                   After
        """

        def _flipud_label(label):
            return [label.flipud(self.img_size)]

        return self.transform_labels(_flipud_label)

    def _get_thickness(self):
        h, w = self.img_size
        step_size = 100
        size = min(h, w) + (max(h, w) - min(h, w)) * 0.5
        return int(size) // step_size + 1

    def _get_font(self):
        """
        The function get size of font for image with given size
        :return: font for drawing
        """
        return sly_font.get_font(font_size=sly_font.get_readable_font_size(self.img_size))

    def _draw_tags(self, bitmap):
        """
        The function draws text labels on bitmap from left to right.
        :param bitmap: target image
        """
        texts = [tag.get_compact_str() for tag in self.img_tags]
        sly_image.draw_text_sequence(
            bitmap,
            texts,
            (0, 0),
            sly_image.CornerAnchorMode.TOP_LEFT,
            font=self._get_font(),
        )

    def draw(
        self,
        bitmap: np.ndarray,
        color: Optional[List[int, int, int]] = None,
        thickness: Optional[int] = 1,
        draw_tags: Optional[bool] = False,
        fill_rectangles: Optional[bool] = True,
        draw_class_names: Optional[bool] = False,
    ) -> None:
        """
        Draws current Annotation on image. Modifies mask.

        :param bitmap: Image.
        :type bitmap: np.ndarray
        :param color: Drawing color in :class:`[R, G, B]`.
        :type color: List[int, int, int], optional
        :param thickness: Thickness of the drawing figure.
        :type thickness: int, optional
        :param draw_tags: Determines whether to draw tags on bitmap or not.
        :type draw_tags: bool, optional
        :param fill_rectangles: Choose False if you want to draw only contours of bboxes. By default, True.
        :type fill_rectangles: int, optional
        :param draw_class_names: Determines whether to draw class names on bitmap or not.
        :type draw_class_names: int, optional

        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image and annotation from API
            project_id = 888
            image_id = 555555

            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_info = api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            img = api.image.download_np(image_id)

            # Draw Annotation on image
            ann.draw(img)

        .. image:: https://i.imgur.com/1W1Nfl1.jpg
            :width: 600
            :height: 500
        """
        tags_font = None
        if draw_tags is True:
            tags_font = self._get_font()
        for label in self._labels:
            if not fill_rectangles and isinstance(label.geometry, Rectangle):
                label.draw_contour(
                    bitmap,
                    color=color,
                    thickness=thickness,
                    draw_tags=draw_tags,
                    tags_font=tags_font,
                    draw_class_name=draw_class_names,
                    class_name_font=tags_font,
                )
                continue
            label.draw(
                bitmap,
                color=color,
                thickness=thickness,
                draw_tags=draw_tags,
                tags_font=tags_font,
                draw_class_name=draw_class_names,
                class_name_font=tags_font,
            )
        if draw_tags:
            self._draw_tags(bitmap)

    def draw_contour(
        self,
        bitmap: np.ndarray,
        color: Optional[List[int, int, int]] = None,
        thickness: Optional[int] = 1,
        draw_tags: Optional[bool] = False,
    ) -> None:
        """
        Draws geometry contour of Annotation on image. Modifies mask.

        :param bitmap: Image.
        :type bitmap: np.ndarray
        :param color: Drawing color in :class:`[R, G, B]`.
        :type color: List[int, int, int], optional
        :param thickness: Thickness of the drawing figure.
        :type thickness: int, optional
        :param draw_tags: Determines whether to draw tags on bitmap or not.
        :type draw_tags: bool, optional
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image and annotation from API
            project_id = 888
            image_id = 555555

            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_info = api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            img = api.image.download_np(image_id)

            # Draw Annotation contour on image
            ann.draw_contour(img)

        .. image:: https://i.imgur.com/F8KGZS4.jpg
            :width: 600
            :height: 500
        """
        tags_font = None
        if draw_tags is True:
            tags_font = self._get_font()
        for label in self._labels:
            label.draw_contour(
                bitmap,
                color=color,
                thickness=thickness,
                draw_tags=draw_tags,
                tags_font=tags_font,
            )
        if draw_tags:
            self._draw_tags(bitmap)

    @classmethod
    def from_img_path(cls, img_path: str) -> Annotation:
        """
        Creates empty Annotation from image.

        :param img_path: Path to the input image.
        :type img_path: str
        :return: Annotation
        :rtype: :class:`Annotation<Annotation>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            img_path = "/home/admin/work/docs/my_dataset/img/example.jpeg"
            ann = sly.Annotation.from_img_path(img_path)
        """
        img = sly_image.read(img_path)
        img_size = img.shape[:2]
        return cls(img_size)

    @classmethod
    def stat_area(
        cls, render: np.ndarray, names: List[str], colors: List[List[int, int, int]]
    ) -> Dict[str, float]:
        """
        Get statistics about color area representation on the given render for the current Annotation.

        :param render: Target render.
        :type render: np.ndarray
        :param names: List of color names.
        :type names: List[str]
        :param colors: List of :class:`[R, G, B]` colors.
        :type colors: List[List[int, int, int]]
        :return: Colors area representation on the given render
        :rtype: :class:`dict`
        :raises: :class:`RuntimeError` if len(names) != len(colors)

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image and annotation from API
            project_id = 888
            image_id = 555555

            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_info = api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            img = api.image.download_np(image_id)

            class_names = []
            class_colors = []
            for label in ann.labels:
                class_names.append(label.obj_class.name)
                class_colors.append(label.obj_class.color)

            ann.draw_pretty(img, thickness=3)

            ann_stats = ann.stat_area(img, class_names, class_colors)
            print(ann_stats)
            # Output: {
            #     "lemon":0.45548266166822865,
            #     "kiwi":0.5697047797563262,
            #     "unlabeled":98.97481255857544,
            #     "height":800,
            #     "width":1067,
            #     "channels":3
            # }

            print(stat_area)
        """
        if len(names) != len(colors):
            raise RuntimeError(
                "len(names) != len(colors) [{} != {}]".format(len(names), len(colors))
            )

        result = {}

        height, width = render.shape[:2]
        total_pixels = height * width

        channels = None
        if len(render.shape) == 2:
            channels = 1
        elif len(render.shape) == 3:
            channels = render.shape[2]

        unlabeled_done = False

        covered_pixels = 0
        for name, color in zip(names, colors):
            col_name = name
            if name == "unlabeled":
                unlabeled_done = True
            class_mask = np.all(render == color, axis=-1).astype("uint8")
            cnt_pixels = class_mask.sum()
            covered_pixels += cnt_pixels
            result[col_name] = cnt_pixels / total_pixels * 100.0

        if covered_pixels > total_pixels:
            raise RuntimeError("Class colors mistake: covered_pixels > total_pixels")

        if unlabeled_done is False:
            result["unlabeled"] = (total_pixels - covered_pixels) / total_pixels * 100.0

        result["height"] = height
        result["width"] = width
        result["channels"] = channels
        return result

    def stat_class_count(self, class_names: Optional[List[str]] = None) -> defaultdict:
        """
        Get statistics about number of each class in Annotation.

        :param class_names: List of classes names.
        :type class_names: List[str], optional
        :return: Number of each class in Annotation and total number of classes
        :rtype: :class:`defaultdict`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            # Create object classes
            class_kiwi = sly.ObjClass('kiwi', sly.Rectangle)
            class_lemon = sly.ObjClass('lemon', sly.Rectangle)

            # Create labels
            label_kiwi = sly.Label(sly.Rectangle(100, 100, 700, 900), class_kiwi)
            label_lemon = sly.Label(sly.Rectangle(200, 200, 500, 600), class_lemon)
            labels_arr = [label_kiwi, label_lemon]

            # Create annotation
            height, width = 300, 400
            ann = sly.Annotation((height, width), labels_arr)

            stat_class = ann.stat_class_count(['lemon', 'kiwi'])

            # Output: defaultdict(<class 'int'>, {'lemon': 1, 'kiwi': 1, 'total': 2})
        """
        total = 0
        stat = {name: 0 for name in class_names}
        for label in self._labels:
            cur_name = label.obj_class.name
            if cur_name not in stat:
                raise KeyError("Class {!r} not found in {}".format(cur_name, class_names))
            stat[cur_name] += 1
            total += 1
        stat["total"] = total
        return stat

    def draw_class_idx_rgb(self, render: np.ndarray, name_to_index: Dict[str, int]) -> None:
        """
        Draws current Annotation on render.

        :param render: Target render to draw classes.
        :type render: np.ndarray
        :param name_to_index: Dict where keys are class names and values are class indices to draw on render.
        :type name_to_index: dict

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image and annotation from API
            project_id = 888
            image_id = 555555

            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_info = api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            img = api.image.download_np(image_id)

            # Draw Annotation on image
            name_to_index = {'lemon': 90, 'kiwi': 195}
            ann.draw_class_idx_rgb(img, name_to_index)

        .. image:: https://i.imgur.com/ACSaBgw.jpg
            :width: 600
            :height: 500
        """
        for label in self._labels:
            class_idx = name_to_index[label.obj_class.name]
            color = [class_idx, class_idx, class_idx]
            label.draw(render, color=color, thickness=1)

    @property
    def custom_data(self):
        """custom_data"""
        return self._custom_data.copy()

    def filter_labels_by_min_side(
        self,
        thresh: int,
        filter_operator: Optional[Callable] = operator.lt,  # operator from the operator module
        classes: Optional[List[str]] = None,
    ) -> Annotation:
        """
        Filters Labels of the current Annotation by side. If minimal side is smaller than Label threshold it will be ignored.

        :param thresh: Side threshold to filter.
        :type thresh: int
        :param filter_operator: Type of filter operation.
        :type filter_operator: operator
        :param classes: List of Labels names to apply filter.
        :type classes: List[str]
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image and annotation from API
            project_id = 888
            image_id = 555555

            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_info = api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            img = api.image.download_np(image_id)

            # Filter Labels
            # Remember that Annotation object is immutable, and we need to assign new instance of Annotation to a new variable
            filtered_ann = ann.filter_labels_by_min_side(200)

            # Draw filtered Annotation on image
            filtered_ann.draw(img)

        .. list-table::

            * - .. figure:: https://i.imgur.com/6huO1se.jpg

                   Before

              - .. figure:: https://i.imgur.com/uunTbPR.jpg

                   After
        """

        def filter(label):
            if classes == None or label.obj_class.name in classes:
                bbox = label.geometry.to_bbox()
                height_px = bbox.height
                width_px = bbox.width
                if filter_operator(min(height_px, width_px), thresh):
                    return []  # action 'delete'
            return [label]

        return self.transform_labels(filter)

    def get_label_by_id(self, sly_id: int) -> Union[Label, None]:
        """
        Get Label from current Annotation by sly_id.

        :param sly_id: Label ID from Supervisely server.
        :type sly_id: int
        :return: Label or None
        :rtype: :class:`Label<supervisely.annotation.label.Label>` or :class:`NoneType`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # To get Label ID you must first access ProjectMeta
            PROJECT_ID = 999

            meta_json = api.project.get_meta(PROJECT_ID)
            meta = sly.ProjectMeta.from_json(meta_json)

            # Get desired image id to which label belongs to download annotation
            image_id = 376728
            ann_info = api.annotation.download(image_id)
            ann_json = ann_info.annotation
            ann = sly.Annotation.from_json(ann_json, meta)

            # Get Label by it's ID
            label_by_id = ann.get_label_by_id(sly_id=2263842)
            print(label_by_id.to_json())
            # Output: {
            #     "classTitle":"kiwi",
            #     "description":"",
            #     "tags":[],
            #     "points":{
            #         "exterior":[
            #             [481, 549],
            #             [641, 703]
            #         ],
            #         "interior":[]
            #     },
            #     "labelerLogin":"cxnt",
            #     "updatedAt":"2020-12-11T08:11:43.249Z",
            #     "createdAt":"2020-12-10T09:38:57.969Z",
            #     "id":2263842,
            #     "classId":7370,
            #     "geometryType":"rectangle",
            #     "shape":"rectangle"
            # }

            # Returns None if Label ID doesn't exist on the given image ID
            label_by_id = ann.get_label_by_id(sly_id=9999999)
            # Output: None
        """
        for label in self._labels:
            if label.geometry.sly_id == sly_id:
                return label
        return None

    def merge(self, other: Annotation) -> Annotation:
        """
        Merge current Annotation with another Annotation.

        :param other: Annotation to merge.
        :type other: Annotation
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

         .. code-block:: python

            import supervisely as sly

            # Create annotation
            meta_lemon = sly.TagMeta('lemon_tag', sly.TagValueType.ANY_STRING)
            tag_lemon = sly.Tag(meta_lemon, 'lemon')
            tags_lemon = sly.TagCollection([tag_lemon])
            class_lemon = sly.ObjClass('lemon', sly.Rectangle)
            label_lemon = sly.Label(sly.Rectangle(100, 100, 200, 200), class_lemon)
            height, width = 300, 400
            ann_lemon = sly.Annotation((height, width), [label_lemon], tags_lemon)

            # Create annotation to merge
            meta_kiwi= sly.TagMeta('kiwi_tag', sly.TagValueType.ANY_STRING)
            tag_kiwi = sly.Tag(meta_kiwi, 'kiwi')
            tags_kiwi = sly.TagCollection([tag_kiwi])
            class_kiwi = sly.ObjClass('kiwi', sly.Rectangle)
            label_kiwi = sly.Label(sly.Rectangle(200, 100, 700, 200), class_kiwi)
            height, width = 700, 500
            ann_kiwi = sly.Annotation((height, width), [label_kiwi], tags_kiwi)

            # Merge annotations
            ann_merge = ann_lemon.merge(ann_kiwi)

            for label in ann_merge.labels:
                print(label.obj_class.name)

            # Output: lemon
            # Output: kiwi
        """
        res = self.clone()
        res = res.add_labels(other.labels)
        res = res.add_tags(other.img_tags)
        return res

    def draw_pretty(
        self,
        bitmap: np.ndarray,
        color: Optional[List[int, int, int]] = None,
        thickness: Optional[int] = None,
        opacity: Optional[float] = 0.5,
        draw_tags: Optional[bool] = False,
        output_path: Optional[str] = None,
        fill_rectangles: Optional[bool] = True,
    ) -> None:
        """
        Draws current Annotation on image with contour. Modifies mask.

        :param bitmap: Image.
        :type bitmap: np.ndarray
        :param color: Drawing color in :class:`[R, G, B]`.
        :type color: List[int, int, int], optional
        :param thickness: Thickness of the drawing figure.
        :type thickness: int, optional
        :param opacity: Opacity of the drawing figure.
        :type opacity: float, optional
        :param draw_tags: Determines whether to draw tags on bitmap or not.
        :type draw_tags: bool, optional
        :param output_path: Saves modified image to the given path.
        :type output_path: str, optional
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image and annotation from API
            project_id = 888
            image_id = 555555

            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_info = api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            img = api.image.download_np(image_id)

            # Draw pretty Annotation on image
            ann.draw_pretty(img, thickness=3)

        .. image:: https://i.imgur.com/6huO1se.jpg
            :width: 600
            :height: 500
        """
        if thickness is None:
            thickness = self._get_thickness()
        height, width = bitmap.shape[:2]
        vis_filled = np.zeros((height, width, 3), np.uint8)
        self.draw(
            vis_filled,
            color=color,
            thickness=thickness,
            draw_tags=draw_tags,
            fill_rectangles=fill_rectangles,
        )
        non_empty_pixels = np.tile(np.any(vis_filled != 0, axis=2)[:, :, np.newaxis], (1, 1, 3))
        mixes_bitmap = np.where(
            non_empty_pixels, vis_filled * opacity + bitmap * (1 - opacity), bitmap
        ).astype(np.uint8)
        np.copyto(bitmap, mixes_bitmap)
        if thickness > 0:
            self.draw_contour(bitmap, color=color, thickness=thickness, draw_tags=draw_tags)
        if output_path:
            sly_image.write(output_path, bitmap)

    def to_nonoverlapping_masks(self, mapping: Dict[ObjClass, ObjClass]) -> Annotation:
        """
        Create new annotation with non-overlapping labels masks. Convert classes to Bitmap or skip them.

        :param mapping: Dict with ObjClasses for mapping.
        :type mapping: Dict[ObjClass, ObjClass]
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

        .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image annotation from API
            project_id = 7548
            image_id = 2254937
            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)
            ann_info = api.annotation.download(image_id)
            print(json.dumps(ann_info.annotation, indent=4))

            # Output: {
            #     "description": "",
            #     "tags": [],
            #     "size": {
            #         "height": 800,
            #         "width": 1067
            #     },
            #     "objects": [
            #         {
            #             "id": 56656282,
            #             "classId": 122357,
            #             "description": "",
            #             "geometryType": "bitmap",
            #             "labelerLogin": "alex",
            #             "createdAt": "2021-10-15T11:01:40.805Z",
            #             "updatedAt": "2021-10-15T11:01:40.805Z",
            #             "tags": [],
            #             "classTitle": "lemon",
            #             "bitmap": {
            #                 "data": "eJwBuwJE/YlQTkcNChoKAAAADUlIRFIAAAE3AAAApgEDAAAAhaFaIwAAAAZQTFRFAAAA...,
            #                 "origin": [
            #                     589,
            #                     372
            #                 ]
            #             }
            #         },
            #         {
            #             "id": 56656281,
            #             "classId": 122356,
            #             "description": "",
            #             "geometryType": "polygon",
            #             "labelerLogin": "alex",
            #             "createdAt": "2021-10-15T13:22:58.178Z",
            #             "updatedAt": "2021-10-15T13:22:58.178Z",
            #             "tags": [],
            #             "classTitle": "kiwi",
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         719,
            #                         115
            #                     ],
            #                   ...
            #                     [
            #                         732,
            #                         123
            #                     ]
            #                 ],
            #                 "interior": []
            #             }
            #         },
            #         {
            #             "id": 56656280,
            #             "classId": 122356,
            #             "description": "",
            #             "geometryType": "polygon",
            #             "labelerLogin": "alex",
            #             "createdAt": "2021-10-15T13:22:58.178Z",
            #             "updatedAt": "2021-10-15T13:22:58.178Z",
            #             "tags": [],
            #             "classTitle": "kiwi",
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         250,
            #                         216
            #                     ],
            #                   ...
            #                     [
            #                         278,
            #                         212
            #                     ]
            #                 ],
            #                 "interior": []
            #             }
            #         },
            #         {
            #             "id": 56656279,
            #             "classId": 122356,
            #             "description": "",
            #             "geometryType": "polygon",
            #             "labelerLogin": "alex",
            #             "createdAt": "2021-10-15T13:22:58.177Z",
            #             "updatedAt": "2021-10-15T13:22:58.177Z",
            #             "tags": [],
            #             "classTitle": "kiwi",
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         554,
            #                         581
            #                     ],
            #                   ...
            #                     [
            #                         560,
            #                         587
            #                     ]
            #                 ],
            #                 "interior": []
            #             }
            #         }
            #     ]
            # }

            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            # Create mapping. Let's check 'kiwi' classes and skip 'lemon' classes.
            mapping = {}
            for label in ann.labels:
                if label.obj_class.name not in mapping:
                    if label.obj_class.name == 'lemon':
                        mapping[label.obj_class] = None
                    else:
                        new_obj_class = sly.ObjClass(label.obj_class.name, Bitmap)
                        mapping[label.obj_class] = new_obj_class
            nonoverlap_ann = ann.to_nonoverlapping_masks(mapping)
            print(json.dumps(nonoverlap_ann.to_json(), indent=4))

            # Output: {
            #     "description": "",
            #     "size": {
            #         "height": 800,
            #         "width": 1067
            #     },
            #     "tags": [],
            #     "objects": [
            #         {
            #             "classTitle": "kiwi",
            #             "description": "",
            #             "tags": [],
            #             "bitmap": {
            #                 "origin": [
            #                     187,
            #                     396
            #                 ],
            #                 "data": "eJwBLALT/YlQTkcNChoKAAAADUlIRFIAAACuAAAAzgEDAAAAnTar9wAAAAZQTFRFAAAA...
            #             },
            #             "shape": "bitmap",
            #             "geometryType": "bitmap"
            #         },
            #         {
            #             "classTitle": "kiwi",
            #             "description": "",
            #             "tags": [],
            #             "bitmap": {
            #                 "origin": [
            #                     365,
            #                     385
            #                 ],
            #                 "data": "eJwB4gEd/olQTkcNChoKAAAADUlIRFIAAACbAAAAwgEDAAAAZC4i8AAAAAZQTFRFAAAA...
            #             },
            #             "shape": "bitmap",
            #             "geometryType": "bitmap"
            #         },
            #         {
            #             "classTitle": "kiwi",
            #             "description": "",
            #             "tags": [],
            #             "bitmap": {
            #                 "origin": [
            #                     469,
            #                     506
            #                 ],
            #                 "data": "eJwBHgLh/YlQTkcNChoKAAAADUlIRFIAAAC1AAAArQEDAAAAzBisHAAAAAZQTFRFAAAA...
            #             },
            #             "shape": "bitmap",
            #             "geometryType": "bitmap"
            #         }
            #     ],
            #     "customBigData": {}
            # }
        """
        common_img = np.zeros(self.img_size, np.int32)  # size is (h, w)
        for idx, lbl in enumerate(self.labels, start=1):
            # if mapping[lbl.obj_class] is not None:
            if isinstance(lbl.geometry, (Bitmap, Polygon)):
                lbl.draw(common_img, color=idx)

        # (unique, counts) = np.unique(common_img, return_counts=True)
        new_labels = []
        for idx, lbl in enumerate(self.labels, start=1):
            dest_class = mapping[lbl.obj_class]
            if dest_class is None:
                continue  # skip labels

            mask = common_img == idx

            if np.any(mask):  # figure may be entirely covered by others
                g = lbl.geometry
                new_mask = Bitmap(data=mask, extra_validation=False)
                new_lbl = lbl.clone(geometry=new_mask, obj_class=dest_class)
                new_labels.append(new_lbl)
        new_ann = self.clone(labels=new_labels)
        return new_ann

    def to_indexed_color_mask(
        self,
        mask_path: str,
        palette: Optional[Image.ADAPTIVE] = Image.ADAPTIVE,  # pylint: disable=no-member
        colors: Optional[int] = 256,
    ) -> None:
        """
        Draw current Annotation on image and save it in PIL format.

        :param mask_path: Saves image to the given path.
        :type mask_path: str
        :param palette: Palette to use when converting image from mode "RGB" to "P".
        :type palette: Available palettes are WEB or ADAPTIVE, optional
        :param colors: Number of colors to use for the ADAPTIVE palette.
        :type colors: int, optional
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`
        """
        mask = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        for label in self.labels:
            label.geometry.draw(mask, label.obj_class.color)

        im = Image.fromarray(mask)
        im = im.convert("P", palette=palette, colors=colors)

        ensure_base_path(mask_path)
        im.save(mask_path)

    def add_bg_object(self, bg_obj_class: ObjClass):
        """add_bg_object"""
        if bg_obj_class not in [label.obj_class for label in self.labels]:
            bg_geometry = Rectangle.from_size(self.img_size)
            bg_geometry = bg_geometry.convert(new_geometry=bg_obj_class.geometry_type)[0]

            new_label = Label(bg_geometry, bg_obj_class)

            updated_labels = self.labels
            updated_labels.insert(0, new_label)

            return self.clone(labels=updated_labels)
        else:
            return self

    def to_segmentation_task(self) -> Annotation:
        """
        Convert Annotation classes by joining labels with same object classes to one label. Applies to Bitmap only.

        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image annotation from API
            project_id = 7473
            image_id = 2223200
            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)
            ann_info = api.annotation.download(image_id)
            print(json.dumps(ann_info.annotation, indent=4))

            # Output: {
            #     "description": "",
            #     "tags": [],
            #     "size": {
            #         "height": 800,
            #         "width": 1067
            #     },
            #     "objects": [
            #         {
            #             "id": 57388829,
            #             "classId": 121405,
            #             "description": "",
            #             "geometryType": "bitmap",
            #             "labelerLogin": "alex",
            #             "createdAt": "2022-01-02T08:06:05.183Z",
            #             "updatedAt": "2022-01-02T08:07:12.219Z",
            #             "tags": [],
            #             "classTitle": "kiwi",
            #             "bitmap": {
            #                 "data": "eJyNlWs4lHkYxv/z8qp5NZuYSZes1KZm6CDSTuP0mh2ZWYdmULakdBUzkRqRrDG8M1vZZg8W...,
            #                 "origin": [
            #                     481,
            #                     543
            #                 ]
            #             }
            #         },
            #         {
            #             "id": 57388831,
            #             "classId": 121404,
            #             "description": "",
            #             "geometryType": "bitmap",
            #             "labelerLogin": "alex",
            #             "createdAt": "2022-01-02T08:06:59.133Z",
            #             "updatedAt": "2022-01-02T08:07:12.219Z",
            #             "tags": [],
            #             "classTitle": "lemon",
            #             "bitmap": {
            #                 "data": "eJwdV388k/sXfzwmz3TVNm3l94z5saQoXFSG+c26tM1GYjWUISWKK201SchvIcq30tX2rD...,
            #                 "origin": [
            #                     523,
            #                     119
            #                 ]
            #             }
            #         },
            #         {
            #             "id": 57388832,
            #             "classId": 121405,
            #             "description": "",
            #             "geometryType": "bitmap",
            #             "labelerLogin": "alex",
            #             "createdAt": "2022-01-02T08:07:12.104Z",
            #             "updatedAt": "2022-01-02T08:07:12.104Z",
            #             "tags": [],
            #             "classTitle": "kiwi",
            #             "bitmap": {
            #                 "data": "eJw1VglQU8kWTWISHzH6AzwwIEsSCBr2AApBloAJhB0eCEIYDJFN2WHACC7sRghbIAiIMuKj...,
            #                 "origin": [
            #                     773,
            #                     391
            #                 ]
            #             }
            #         }
            #     ]
            # }

            ann = sly.Annotation.from_json(ann_info.annotation, meta)
            segm_ann = ann.to_segmentation_task()
            print(json.dumps(segm_ann.to_json(), indent=4))

            # Output: {
            #     "description": "",
            #     "size": {
            #         "height": 800,
            #         "width": 1067
            #     },
            #     "tags": [],
            #     "objects": [
            #         {
            #             "classTitle": "kiwi",
            #             "description": "",
            #             "tags": [],
            #             "bitmap": {
            #                 "origin": [
            #                     481,
            #                     391
            #                 ],
            #                 "data": "eJwBagSV+4lQTkcNChoKAAAADUlIRFIAAAHpAAABOQEDAAAAjj5K+wAAAAZQTFRFAAAA...
            #             },
            #             "shape": "bitmap",
            #             "geometryType": "bitmap"
            #         },
            #         {
            #             "classTitle": "lemon",
            #             "description": "",
            #             "tags": [],
            #             "bitmap": {
            #                 "origin": [
            #                     523,
            #                     119
            #                 ],
            #                 "data": "eJwBOAPH/IlQTkcNChoKAAAADUlIRFIAAAEsAAABCQEDAAAAFNKIswAAAAZQTFRFAAAA...
            #             },
            #             "shape": "bitmap",
            #             "geometryType": "bitmap"
            #         }
            #     ],
            #     "customBigData": {}
            # }
        """
        class_mask = {}
        for label in self.labels:
            if label.obj_class not in class_mask:
                class_mask[label.obj_class] = np.zeros(self.img_size, np.uint8)
            label.draw(class_mask[label.obj_class], color=255)
        new_labels = []
        for obj_class, white_mask in class_mask.items():
            mask = white_mask == 255
            bitmap = Bitmap(data=mask, extra_validation=False)
            new_labels.append(Label(geometry=bitmap, obj_class=obj_class))
        return self.clone(labels=new_labels)

    def to_detection_task(self, mapping: Dict[ObjClass, ObjClass]) -> Annotation:
        """
        Convert Annotation classes geometries according to mapping dict and checking nonoverlapping masks.
        Converting possible only to Bitmap or Rectangle.

        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image annotation from API
            project_id = 7548
            image_id = 2254942
            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)
            ann_info = api.annotation.download(image_id)
            print(json.dumps(ann_info.annotation, indent=4))

            # Output: {
            #     "description": "",
            #     "tags": [],
            #     "size": {
            #         "height": 800,
            #         "width": 1067
            #     },
            #     "objects": [
            #         {
            #             "id": 56656282,
            #             "classId": 122357,
            #             "description": "",
            #             "geometryType": "bitmap",
            #             "labelerLogin": "alex",
            #             "createdAt": "2021-10-15T11:01:40.805Z",
            #             "updatedAt": "2021-10-15T11:01:40.805Z",
            #             "tags": [],
            #             "classTitle": "lemon",
            #             "bitmap": {
            #                 "data": "eJwBuwJE/YlQTkcNChoKAAAADUlIRFIAAAE3AAAApgEDAAAAhaFaIwAAAAZQTFRFAAAA...,
            #                 "origin": [
            #                     589,
            #                     372
            #                 ]
            #             }
            #         },
            #         {
            #             "id": 56656281,
            #             "classId": 122356,
            #             "description": "",
            #             "geometryType": "polygon",
            #             "labelerLogin": "alex",
            #             "createdAt": "2021-10-15T13:22:58.178Z",
            #             "updatedAt": "2021-10-15T13:22:58.178Z",
            #             "tags": [],
            #             "classTitle": "kiwi",
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         719,
            #                         115
            #                     ],
            #                   ...
            #                     [
            #                         732,
            #                         123
            #                     ]
            #                 ],
            #                 "interior": []
            #             }
            #         },
            #         {
            #             "id": 56656280,
            #             "classId": 122356,
            #             "description": "",
            #             "geometryType": "polygon",
            #             "labelerLogin": "alex",
            #             "createdAt": "2021-10-15T13:22:58.178Z",
            #             "updatedAt": "2021-10-15T13:22:58.178Z",
            #             "tags": [],
            #             "classTitle": "kiwi",
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         250,
            #                         216
            #                     ],
            #                   ...
            #                     [
            #                         278,
            #                         212
            #                     ]
            #                 ],
            #                 "interior": []
            #             }
            #         },
            #         {
            #             "id": 56656279,
            #             "classId": 122356,
            #             "description": "",
            #             "geometryType": "polygon",
            #             "labelerLogin": "alex",
            #             "createdAt": "2021-10-15T13:22:58.177Z",
            #             "updatedAt": "2021-10-15T13:22:58.177Z",
            #             "tags": [],
            #             "classTitle": "kiwi",
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         554,
            #                         581
            #                     ],
            #                   ...
            #                     [
            #                         560,
            #                         587
            #                     ]
            #                 ],
            #                 "interior": []
            #             }
            #         }
            #     ]
            # }

            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            # Create mapping for classes converting. Let's convert classes to Rectangle.
            mapping = {}
            for label in ann.labels:
                if label.obj_class.name not in mapping:
                    new_obj_class = sly.ObjClass(label.obj_class.name, Rectangle)
                    mapping[label.obj_class] = new_obj_class

            det_ann = ann.to_detection_task(mapping)
            print(json.dumps(det_ann.to_json(), indent=4))

            # Output: {
            #     "description": "",
            #     "size": {
            #         "height": 800,
            #         "width": 1067
            #     },
            #     "tags": [],
            #     "objects": [
            #         {
            #             "classTitle": "lemon",
            #             "description": "",
            #             "tags": [],
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         589,
            #                         372
            #                     ],
            #                     [
            #                         899,
            #                         537
            #                     ]
            #                 ],
            #                 "interior": []
            #             },
            #             "geometryType": "rectangle",
            #             "shape": "rectangle"
            #         },
            #         {
            #             "classTitle": "kiwi",
            #             "description": "",
            #             "tags": [],
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         612,
            #                         110
            #                     ],
            #                     [
            #                         765,
            #                         282
            #                     ]
            #                 ],
            #                 "interior": []
            #             },
            #             "geometryType": "rectangle",
            #             "shape": "rectangle"
            #         },
            #         {
            #             "classTitle": "kiwi",
            #             "description": "",
            #             "tags": [],
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         196,
            #                         212
            #                     ],
            #                     [
            #                         352,
            #                         380
            #                     ]
            #                 ],
            #                 "interior": []
            #             },
            #             "geometryType": "rectangle",
            #             "shape": "rectangle"
            #         },
            #         {
            #             "classTitle": "kiwi",
            #             "description": "",
            #             "tags": [],
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         425,
            #                         561
            #                     ],
            #                     [
            #                         576,
            #                         705
            #                     ]
            #                 ],
            #                 "interior": []
            #             },
            #             "geometryType": "rectangle",
            #             "shape": "rectangle"
            #         }
            #     ],
            #     "customBigData": {}
            # }
        """
        aux_mapping = mapping.copy()

        to_render_mapping = {}
        to_render_labels = []
        other_labels = []
        _polygons_to_bitmaps_classes = {}
        for lbl in self.labels:
            if type(lbl.geometry) in [Bitmap, Polygon]:
                to_render_labels.append(lbl)
                if type(lbl.geometry) is Polygon:
                    new_class = _polygons_to_bitmaps_classes.get(lbl.obj_class.name, None)
                    if new_class is None:
                        new_class = lbl.obj_class.clone(geometry_type=Bitmap)
                        _polygons_to_bitmaps_classes[lbl.obj_class.name] = new_class
                        aux_mapping[new_class] = aux_mapping[lbl.obj_class]
                    to_render_mapping[lbl.obj_class] = new_class

                else:
                    to_render_mapping[lbl.obj_class] = lbl.obj_class
            else:
                other_labels.append(lbl)
        ann_raster = self.clone(labels=to_render_labels)
        ann_raster = ann_raster.to_nonoverlapping_masks(to_render_mapping)
        new_labels = []
        for lbl in [*ann_raster.labels, *other_labels]:
            dest_class = aux_mapping[lbl.obj_class]
            if dest_class is None:
                continue  # skip labels
            if dest_class == lbl.obj_class and lbl.obj_class.geometry_type != AnyGeometry:
                new_labels.append(lbl)
            else:
                bbox = lbl.geometry.to_bbox()
                new_lbl = lbl.clone(geometry=bbox, obj_class=dest_class)
                new_labels.append(new_lbl)
        new_ann = self.clone(labels=new_labels)
        return new_ann

    def masks_to_imgaug(
        self, class_to_index: Dict[str, int]
    ) -> Union[SegmentationMapsOnImage, None]:
        """
        Convert current annotation objects masks to SegmentationMapsOnImage format.

        :param class_to_index: Dict matching class name to index.
        :type class_to_index: dict
        :return: SegmentationMapsOnImage, otherwise :class:`None`
        :rtype: :class:`SegmentationMapsOnImage` or :class:`NoneType`
        """
        try:
            from imgaug.augmentables.segmaps import SegmentationMapsOnImage
        except ModuleNotFoundError as e:
            logger.error(
                f'{e}. Try to install extra dependencies. Run "pip install supervisely[aug]"'
            )
            raise e

        h = self.img_size[0]
        w = self.img_size[1]
        mask = np.zeros((h, w, 1), dtype=np.int32)

        for index, label in enumerate(self.labels, start=1):
            label: Label
            if type(label.geometry) == Bitmap:
                if class_to_index is not None:
                    label.draw(mask, class_to_index[label.obj_class.name])
                else:
                    label.draw(mask, index)

        segmaps = None
        if np.any(mask):
            segmaps = SegmentationMapsOnImage(mask, shape=self.img_size)
        return segmaps

    def bboxes_to_imgaug(self) -> Union[BoundingBoxesOnImage, None]:
        """
        Convert current annotation objects boxes to BoundingBoxesOnImage format.

        :return: BoundingBoxesOnImage, otherwise :class:`None`
        :rtype: :class:`BoundingBoxesOnImage` or :class:`NoneType`
        """
        try:
            from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
        except ModuleNotFoundError as e:
            logger.error(
                f'{e}. Try to install extra dependencies. Run "pip install supervisely[aug]"'
            )
            raise e

        boxes = []
        for label in self.labels:
            if type(label.geometry) == Rectangle:
                rect: Rectangle = label.geometry
                boxes.append(
                    BoundingBox(
                        x1=rect.left,
                        y1=rect.top,
                        x2=rect.right,
                        y2=rect.bottom,
                        label=label.obj_class.name,
                    )
                )
        bbs = None
        if len(boxes) > 0:
            bbs = BoundingBoxesOnImage(boxes, shape=self.img_size)
        return bbs

    @classmethod
    def from_imgaug(
        cls,
        img: np.ndarray,
        ia_boxes: Optional[List[BoundingBoxesOnImage]] = None,
        ia_masks: Optional[List[SegmentationMapsOnImage]] = None,
        index_to_class: Optional[Dict[int, str]] = None,
        meta: Optional[ProjectMeta] = None,
    ) -> Annotation:
        """
        Create Annotation from image and SegmentationMapsOnImage, BoundingBoxesOnImage data or ProjectMeta.

        :param img: Image in numpy format.
        :type img: np.ndarray
        :param ia_boxes: List of BoundingBoxesOnImage data.
        :type ia_boxes: List[BoundingBoxesOnImage], optional
        :param ia_masks:  List of SegmentationMapsOnImage data.
        :type ia_masks: List[SegmentationMapsOnImage], optional
        :param index_to_class: Dictionary specifying index match of class name.
        :type index_to_class: Dict[int, str], optional
        :param meta: ProjectMeta.
        :type meta: ProjectMeta, optional
        :raises: :class:`ValueError`, if ia_boxes or ia_masks and meta is None
        :raises: :class:`KeyError`, if processed ObjClass not found in meta
        :return: Annotation object
        :rtype: :class:`Annotation<Annotation>`
        """
        if ((ia_boxes is not None) or (ia_masks is not None)) and meta is None:
            raise ValueError("Project meta has to be provided")

        labels = []
        if ia_boxes is not None:
            for ia_box in ia_boxes:
                obj_class = meta.get_obj_class(ia_box.label)
                if obj_class is None:
                    raise KeyError("Class {!r} not found in project meta".format(ia_box.label))
                lbl = Label(
                    Rectangle(top=ia_box.y1, left=ia_box.x1, bottom=ia_box.y2, right=ia_box.x2),
                    obj_class,
                )
                labels.append(lbl)

        if ia_masks is not None:
            if index_to_class is None:
                raise ValueError(
                    "mapping from index to class name is needed to transform masks to SLY format"
                )
            class_mask = ia_masks.get_arr()
            # mask = white_mask == 255
            (unique, counts) = np.unique(class_mask, return_counts=True)
            for index, count in zip(unique, counts):
                if index == 0:
                    continue
                mask = class_mask == index
                bitmap = Bitmap(data=mask[:, :, 0])
                restore_class = meta.get_obj_class(index_to_class[index])
                labels.append(Label(geometry=bitmap, obj_class=restore_class))

        return cls(img_size=img.shape[:2], labels=labels)

    def is_empty(self) -> bool:
        """
        Check whether annotation contains labels or tags, or not.

        :returns: True if annotation is empty, False otherwise.
        :rtype: :class:`bool`
        """
        if len(self.labels) == 0 and len(self.img_tags) == 0:
            return True
        else:
            return False

    def filter_labels_by_classes(self, keep_classes: List[str]) -> Annotation:
        """
        Filter annotation labels by given classes names.

        :param keep_classes: List with classes names.
        :type keep_classes: List[str]
        :return: New instance of Annotation
        :rtype: :class:`Annotation<Annotation>`

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image annotation from API
            project_id = 7473
            image_id = 2223200
            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)
            ann_info = api.annotation.download(image_id)
            print(json.dumps(ann_info.annotation, indent=4))

            # Output: {
            #     "description": "",
            #     "tags": [],
            #     "size": {
            #         "height": 800,
            #         "width": 1067
            #     },
            #     "objects": [
            #         {
            #             "id": 57388829,
            #             "classId": 121405,
            #             "description": "",
            #             "geometryType": "bitmap",
            #             "labelerLogin": "alex",
            #             "createdAt": "2022-01-02T08:06:05.183Z",
            #             "updatedAt": "2022-01-02T08:07:12.219Z",
            #             "tags": [],
            #             "classTitle": "kiwi",
            #             "bitmap": {
            #                 "data": "eJyNlWs4lHkYxv/z8qp5NZuYSZes1KZm6CDSTuP0mh2ZWYdmULakdBUzkRqRrDG8M1vZZg8WIy...,
            #                 "origin": [
            #                     481,
            #                     543
            #                 ]
            #             }
            #         },
            #         {
            #             "id": 57388831,
            #             "classId": 121404,
            #             "description": "",
            #             "geometryType": "bitmap",
            #             "labelerLogin": "alex",
            #             "createdAt": "2022-01-02T08:06:59.133Z",
            #             "updatedAt": "2022-01-02T08:07:12.219Z",
            #             "tags": [],
            #             "classTitle": "lemon",
            #             "bitmap": {
            #                 "data": "eJwdV388k/sXfzwmz3TVNm3l94z5saQoXFSG+c26tM1GYjWUISWKK201SchvIcq30tX2rDuTdE...,
            #                 "origin": [
            #                     523,
            #                     119
            #                 ]
            #             }
            #         },
            #         {
            #             "id": 57388832,
            #             "classId": 121405,
            #             "description": "",
            #             "geometryType": "bitmap",
            #             "labelerLogin": "alex",
            #             "createdAt": "2022-01-02T08:07:12.104Z",
            #             "updatedAt": "2022-01-02T08:07:12.104Z",
            #             "tags": [],
            #             "classTitle": "kiwi",
            #             "bitmap": {
            #                 "data": "eJw1VglQU8kWTWISHzH6AzwwIEsSCBr2AApBloAJhB0eCEIYDJFN2WHACC7sRghbIAiIMuKj...,
            #                 "origin": [
            #                     773,
            #                     391
            #                 ]
            #             }
            #         }
            #     ]
            # }

            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            # Let's filter 'lemon' class
            keep_classes = ['lemon']
            filter_ann = ann.filter_labels_by_classes(keep_classes)
            print(json.dumps(filter_ann.to_json(), indent=4))

            # Output: {
            #     "description": "",
            #     "size": {
            #         "height": 800,
            #         "width": 1067
            #     },
            #     "tags": [],
            #     "objects": [
            #         {
            #             "classTitle": "lemon",
            #             "description": "",
            #             "tags": [],
            #             "bitmap": {
            #                 "origin": [
            #                     523,
            #                     119
            #                 ],
            #                 "data": "eJwBOAPH/IlQTkcNChoKAAAADUlIRFIAAAEsAAABCQEDAAAAFNKIswAAAAZQTFRFAAAA...,
            #             },
            #             "shape": "bitmap",
            #             "geometryType": "bitmap",
            #             "labelerLogin": "alex",
            #             "updatedAt": "2022-01-02T08:07:12.219Z",
            #             "createdAt": "2022-01-02T08:06:59.133Z",
            #             "id": 57388831,
            #             "classId": 121404
            #         }
            #     ],
            #     "customBigData": {}
            # }
        """
        new_labels = []
        for lbl in self.labels:
            if lbl.obj_class.name in keep_classes:
                new_labels.append(lbl.clone())
        return self.clone(labels=new_labels)

    def get_bindings(self) -> Dict[str, List[Label]]:
        """Returns dictionary with bindings keys as keys and list of labels as values.

        :return: Dictionary with bindings keys as keys and list of labels as values.
        :rtype: Dict[str, List[Label]]
        """
        d = defaultdict(list)
        for label in self.labels:
            # if label.binding_key is not None:
            d[label.binding_key].append(label)
        return d

    def discard_bindings(self) -> None:
        """Remove binding keys from all labels."""
        for label in self.labels:
            label.binding_key = None

    @classmethod
    def _to_pixel_coordinate_system_json(cls, data: Dict) -> Dict:
        """
        Convert label geometry from subpixel precision to pixel precision.

        In the labeling tool, labels are created with subpixel precision,
        which means that the coordinates of the geometry can have decimal values representing fractions of a pixel.
        However, in Supervisely SDK, geometry coordinates are represented using pixel precision, where the coordinates are integers representing whole pixels.

        :param data: Label in json format.
        :type data: :class:`dict`
        :return: Json data with coordinates converted to pixel coordinate system.
        :rtype: :class:`dict`
        """
        data = deepcopy(data)  # Avoid modifying the original data
        image_size = [
            data[AnnotationJsonFields.IMG_SIZE][AnnotationJsonFields.IMG_SIZE_HEIGHT],
            data[AnnotationJsonFields.IMG_SIZE][AnnotationJsonFields.IMG_SIZE_WIDTH],
        ]
        new_labels = []
        for label in data[AnnotationJsonFields.LABELS]:
            if label[LabelJsonFields.GEOMETRY_TYPE] == Rectangle.geometry_name():
                label = Rectangle._to_pixel_coordinate_system_json(label, image_size)
            else:
                label = Geometry._to_pixel_coordinate_system_json(label, image_size)
            new_labels.append(label)

        data[AnnotationJsonFields.LABELS] = new_labels
        return data

    @classmethod
    def _to_subpixel_coordinate_system_json(cls, data: Dict) -> Dict:
        """
        Convert label geometry from pixel precision to subpixel precision.

        In the labeling tool, labels are created with subpixel precision,
        which means that the coordinates of the geometry can have decimal values representing fractions of a pixel.
        However, in Supervisely SDK, geometry coordinates are represented using pixel precision, where the coordinates are integers representing whole pixels.

        :param data: Label in json format.
        :type data: dict
        :return: Json data with coordinates converted to subpixel coordinate system.
        :rtype: :class:`dict`
        """
        data = deepcopy(data)  # Avoid modifying the original data
        new_labels = []
        for label in data[AnnotationJsonFields.LABELS]:
            if label[LabelJsonFields.GEOMETRY_TYPE] == Rectangle.geometry_name():
                label = Rectangle._to_subpixel_coordinate_system_json(label)
            else:
                label = Geometry._to_subpixel_coordinate_system_json(label)
            new_labels.append(label)

        data[AnnotationJsonFields.LABELS] = new_labels
        return data

    # def _to_subpixel_coordinate_system(self) -> Annotation:
    #     """
    #     Convert all labels in the annotation from pixel precision to subpixel precision by subtracting a subpixel offset from the coordinates.

    #     In the labeling tool, labels are created with subpixel precision,
    #     which means that the coordinates of the geometry can have decimal values representing fractions of a pixel.
    #     However, in Supervisely SDK, geometry coordinates are represented using pixel precision, where the coordinates are integers representing whole pixels.

    #     :return: New instance of Annotation with labels in subpixel precision.
    #     :rtype: :class:`Annotation<Annotation>`
    #     """
    #     new_ann = self.clone()
    #     new_labels = [label._to_subpixel_coordinate_system() for label in new_ann.labels]
    #     new_ann._labels = new_labels
    #     return new_ann

    def to_coco(
        self,
        coco_image_id: int,
        class_mapping: Dict[str, int],
        coco_ann: Optional[Union[Dict, List]] = None,
        last_label_id: Optional[int] = None,
        coco_captions: Optional[Union[Dict, List]] = None,
        last_caption_id: Optional[int] = None,
    ) -> Tuple[List, List]:
        """
        Convert Supervisely annotation to COCO format annotation ("annotations" field).

        :param coco_image_id: Image id in COCO format.
        :type coco_image_id: int
        :param class_mapping: Dictionary that maps class names to class ids.
        :type class_mapping: Dict[str, int]
        :param coco_ann: COCO annotation in dictionary or list format to append new annotations.
        :type coco_ann: Union[Dict, List], optional
        :param last_label_id: Last label id in COCO format to continue counting.
        :type last_label_id: int, optional
        :param coco_captions: COCO captions in dictionary or list format to append new captions.
        :type coco_captions: Union[Dict, List], optional
        :return: Tuple with list of COCO objects and list of COCO captions.
        :rtype: :class:`tuple`


        :Usage example:

         .. code-block:: python

            import supervisely as sly


            coco_instances = dict(
                info=dict(
                    description="COCO dataset converted from Supervisely",
                    url="None",
                    version=str(1.0),
                    year=2025,
                    contributor="Supervisely",
                    date_created="2025-01-01 00:00:00",
                ),
                licenses=[dict(url="None", id=0, name="None")],
                images=[],
                annotations=[],
                categories=get_categories_from_meta(meta),  # [{"supercategory": "lemon", "id": 1, "name": "lemon"}, ...]
            )

            ann = sly.Annotation.from_json(ann_json, meta)
            image_id = 11
            label_id = 222
            class_mapping = {obj_cls.name: idx for idx, obj_cls in enumerate(meta.obj_classes)}

            curr_coco_ann, _ = ann.to_coco(image_id, class_mapping, coco_instances, label_id)
            # or
            # curr_coco_ann, _ = ann.to_coco(image_id, class_mapping, label_id=label_id)
            # coco_instances["annotations"].extend(curr_coco_ann)

            label_id += len(curr_coco_ann)
            image_id += 1
        """

        from supervisely.convert.image.coco.coco_helper import sly_ann_to_coco

        return sly_ann_to_coco(
            ann=self,
            coco_image_id=coco_image_id,
            class_mapping=class_mapping,
            coco_ann=coco_ann,
            last_label_id=last_label_id,
            coco_captions=coco_captions,
            last_caption_id=last_caption_id,
        )

    def to_yolo(
        self,
        class_names: List[str],
        task_type: Literal["detect", "segment", "pose"] = "detect",
    ) -> List[str]:
        """
        Convert Supervisely annotation to YOLO annotation format.
        Returns a list of strings, each string represents one object.

        :param class_names: List of class names.
        :type class_names: List[str]
        :param task_type: Task type, one of "detection", "segmentation", "pose".
        :type task_type: str
        :return: List of objects in YOLO format.
        :rtype: :class:`list`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            ann = sly.Annotation.from_json(ann_json, meta)
            class_names = [obj_cls.name for obj_cls in meta.obj_classes]

            yolo_lines = ann.to_yolo(class_names, task_type="segmentation")
        """

        from supervisely.convert.image.yolo.yolo_helper import sly_ann_to_yolo

        return sly_ann_to_yolo(ann=self, class_names=class_names, task_type=task_type)

    def to_pascal_voc(
        self,
        image_name: str,
    ) -> Tuple[List, List]:
        """
        Convert Supervisely annotation to Pascal VOC format annotation ("annotations" field).

        :param ann: Supervisely annotation.
        :type ann: :class:`Annotation<supervisely.annotation.annotation.Annotation>`
        :param image_name: Image name.
        :type image_name: :class:`str`
        :return: Tuple with xml tree and instance and class masks in PIL.Image format.
        :rtype: :class:`Tuple`

        :Usage example:

        .. code-block:: python

            import supervisely as sly
            from supervisely.convert.image.pascal_voc.pascal_voc_helper import sly_ann_to_pascal_voc

            ann = sly.Annotation.from_json(ann_json, meta)
            xml_tree, instance_mask, class_mask = sly_ann_to_pascal_voc(ann, image_name)
        """

        from supervisely.convert.image.pascal_voc.pascal_voc_helper import (
            sly_ann_to_pascal_voc,
        )

        return sly_ann_to_pascal_voc(self, image_name)

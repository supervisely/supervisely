import os

import cv2

import supervisely as sly


def _get_video_shape(video_path):
    zero_image_name = os.listdir(video_path)[0]
    image_shape = cv2.imread(os.path.join(video_path, zero_image_name)).shape

    return tuple([int(image_shape[1]), int(image_shape[0])])


def _get_obj_id_to_obj_class(ann_data):
    objects_labels = {}  # object_id: obj_class

    for frame_index, frame_data in ann_data.items():
        for curr_id, label in zip(frame_data["ids"], frame_data["labels"]):
            if objects_labels.get(curr_id, None) is None:
                objects_labels[curr_id] = label.obj_class

    return objects_labels


def get_annotation_keeper(ann_data, video_shape, frames_count):
    obj_id_to_object_class = _get_obj_id_to_obj_class(ann_data)

    ann_keeper = AnnotationKeeper(
        video_shape=(video_shape[1], video_shape[0]),
        obj_id_to_object_class=obj_id_to_object_class,
        video_frames_count=frames_count,
    )

    return ann_keeper


class AnnotationKeeper:
    def __init__(self, video_shape, obj_id_to_object_class, video_frames_count):
        self.video_frames_count = video_frames_count

        self.video_shape = video_shape

        self.project = None
        self.dataset = None
        self.meta = None

        # self.key_id_map = KeyIdMap()
        self.object_id_to_video_object = {}

        self.get_video_objects_list(obj_id_to_object_class)

        self.video_object_collection = sly.VideoObjectCollection(
            list(self.object_id_to_video_object.values())
        )

        self.figures = []
        self.frames_list = []
        self.frames_collection = []

    def add_figures_by_frames(self, data):
        for frame_index, frame_data in data.items():
            if len(frame_data["ids"]) > 0:
                self.add_figures_by_frame(
                    labels_on_frame=frame_data["labels"],
                    objects_indexes=frame_data["ids"],
                    frame_index=frame_index,
                )

    def add_figures_by_frame(self, labels_on_frame, objects_indexes, frame_index):
        temp_figures = []

        for i, label in enumerate(labels_on_frame):
            figure = sly.VideoFigure(
                video_object=self.object_id_to_video_object[objects_indexes[i]],
                geometry=label.geometry,
                frame_index=frame_index,
            )

            temp_figures.append(figure)

        self.figures.append(temp_figures)

    def get_annotation(self) -> sly.VideoAnnotation:
        self.get_frames_list()
        self.frames_collection = sly.FrameCollection(self.frames_list)

        video_annotation = sly.VideoAnnotation(
            self.video_shape,
            self.video_frames_count,
            self.video_object_collection,
            self.frames_collection,
        )

        return video_annotation

    def get_unique_objects(self, obj_list):
        unique_objects = []
        for obj in obj_list:
            if obj.name not in [temp_object.name for temp_object in unique_objects]:
                unique_objects.append(obj)

        return unique_objects

    def get_video_objects_list(self, obj_id_to_object_class):
        for object_id, object_class in obj_id_to_object_class.items():
            self.object_id_to_video_object[object_id] = sly.VideoObject(object_class)

    def get_frames_list(self):
        for index, figure in enumerate(self.figures):
            self.frames_list.append(sly.Frame(figure[0].frame_index, figure))

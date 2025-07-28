class BaseTracker:

    def __init__(self):
        self.settings = {}
        self.result_video_annotation = None

    def update(self, frame, detections):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def reset(self):
        self.result_video_annotation = None
    
    def track(self, frames: list, annoations: list):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _annotations_to_video_annotation(self, annotations: list) -> VideoAnnotation:
        """Convert a list of annotations to a VideoAnnotation object."""
        if not annotations:
            return VideoAnnotation()

        video_annotation = VideoAnnotation()
        for annotation in annotations:
            video_annotation.add(annotation)
        
        return video_annotation
    
    def _create_video_annotation(
        frame_to_annotation: dict,
        tracking_results: list,
        frame_shape: tuple,
        cat2obj: dict,
    ):
        img_h, img_w = frame_shape
        video_objects = {}  # track_id -> VideoObject
        frames = []
        for (i, ann), tracks in zip(frame_to_annotation.items(), tracking_results):
            frame_figures = []
            for track in tracks:
                # crop bbox to image size
                dims = np.array([img_w, img_h, img_w, img_h]) - 1
                track[:4] = np.clip(track[:4], 0, dims)
                x1, y1, x2, y2, track_id, conf, cat = track[:7]
                cat = int(cat)
                track_id = int(track_id)
                rect = sly.Rectangle(y1, x1, y2, x2)
                video_object = video_objects.get(track_id)
                if video_object is None:
                    obj_cls = cat2obj[cat]
                    video_object = sly.VideoObject(obj_cls)
                    video_objects[track_id] = video_object
                frame_figures.append(sly.VideoFigure(video_object, rect, i))
            frames.append(sly.Frame(i, frame_figures))

        objects = list(video_objects.values())
        video_ann = sly.VideoAnnotation(
            img_size=frame_shape,
            frames_count=len(frame_to_annotation),
            objects=sly.VideoObjectCollection(objects),
            frames=sly.FrameCollection(frames),
        )
        return video_ann
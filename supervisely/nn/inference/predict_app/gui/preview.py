class Preview:
    def __init__(self, gui: "PredictAppGui"):
        self.gui = gui
        self._preview_dir = os.path.join(self.gui.static_dir, "preview")
        os.makedirs(self._preview_dir, exist_ok=True)
        self._preview_path = os.path.join(self._preview_dir, "preview.jpg")
        self._peview_url = f"/static/preview/preview.jpg"

        self.inference_settings_editor = self.gui.inference_settings
        self.inference_settings_field = Field(
            content=self.inference_settings_editor,
            title="Inference Settings",
            description="Settings for the inference. YAML format.",
        )

        self.preview_button = Button("Preview", icon="zmdi zmdi-eye")
        self.preview_button.disable()

        self.gallery = GridGallery(
            2,
            sync_views=True,
            enable_zoom=True,
            resize_on_zoom=True,
            empty_message="Click 'Preview' to see the model output.",
        )
        self.error_message = Text("Error during preview", status="error")
        self.error_message.hide()
        self.container = Container(
            widgets=[
                self.inference_settings_field,
                self.preview_button,
            ],
            style="width: 100%;",
            direction="vertical",
            gap=30,
        )
        self.flexbox = Container(
            widgets=[
                self.container,
                Container(widgets=[self.error_message, self.gallery], gap=0, style="width: 100%;"),
            ],
            direction="horizontal",
            overflow="wrap",
            fractions=[3, 7],
            gap=40,
        )
        self.card = Card(title="Preview", content=self.flexbox)

        @self.preview_button.click
        def preview_button_click():
            self.run_preview()

    def _get_frame_annotation(
        self, video_info: VideoInfo, frame_index: int, project_meta: ProjectMeta
    ) -> Annotation:
        video_annotation = VideoAnnotation.from_json(
            self.gui.api.video.annotation.download(video_info.id, frame_index),
            project_meta=project_meta,
            key_id_map=KeyIdMap(),
        )
        frame = video_annotation.frames.get(frame_index)
        img_size = (video_info.frame_height, video_info.frame_width)
        if frame is None:
            return Annotation(img_size)
        labels = []
        for figure in frame.figures:
            labels.append(Label(figure.geometry, figure.video_object.obj_class))
        ann = Annotation(img_size, labels=labels)
        return ann

    def run_preview(self) -> Prediction:
        self.error_message.hide()
        self.gallery.clean_up()
        self.gallery.show()
        self.gallery.loading = True
        try:
            items_settings = self.gui.items.get_item_settings()
            if "video_id" in items_settings:
                video_id = items_settings["video_id"]
                video_info = self.gui.api.video.get_info_by_id(video_id)
                video_frame = random.randint(0, video_info.frames_count - 1)
                self.gui.api.video.frame.download_path(
                    video_info.id, video_frame, self._preview_path
                )
                img_url = self._peview_url
                project_meta = ProjectMeta.from_json(
                    self.gui.api.project.get_meta(video_info.project_id)
                )
                input_ann = self._get_frame_annotation(video_info, video_frame, project_meta)
                prediction = self.gui.model.model_api.predict(
                    input=self._preview_path, **self.gui.get_inference_settings()
                )[0]
                output_ann = prediction.annotation
            else:
                if "project_id" in items_settings:
                    project_id = items_settings["project_id"]
                    dataset_infos = self.gui.api.dataset.get_list(project_id, recursive=True)
                    dataset_infos = [ds for ds in dataset_infos if ds.items_count > 0]
                    if not dataset_infos:
                        raise ValueError("No datasets with items found in the project.")
                    dataset_info = random.choice(dataset_infos)
                elif "dataset_ids" in items_settings:
                    dataset_ids = items_settings["dataset_ids"]
                    dataset_infos = [
                        self.gui.api.dataset.get_info_by_id(dataset_id)
                        for dataset_id in dataset_ids
                    ]
                    dataset_infos = [ds for ds in dataset_infos if ds.items_count > 0]
                    if not dataset_infos:
                        raise ValueError("No items in selected datasets.")
                    dataset_info = random.choice(dataset_infos)
                else:
                    raise ValueError("No valid item settings found for preview.")
                images = self.gui.api.image.get_list(dataset_info.id)
                image_info = random.choice(images)
                img_url = image_info.preview_url

                # @TODO: check infinite preview loading
                # self.gui.api.image.download_path(image_info.id, self._preview_path)
                # img_url = self._peview_url

                project_meta = ProjectMeta.from_json(
                    self.gui.api.project.get_meta(dataset_info.project_id)
                )
                input_ann = Annotation.from_json(
                    self.gui.api.annotation.download(image_info.id).annotation,
                    project_meta=project_meta,
                )
                prediction = self.gui.model.model_api.predict(
                    image_id=image_info.id, **self.gui.get_inference_settings()
                )[0]
                output_ann = prediction.annotation

            self.gallery.append(img_url, input_ann, "Input")
            self.gallery.append(img_url, output_ann, "Output")
            self.error_message.hide()
            self.gallery.show()
            return prediction
        except Exception as e:
            self.gallery.hide()
            self.error_message.text = f"Error during preview: {str(e)}"
            self.error_message.show()
            self.gallery.clean_up()
        finally:
            self.gallery.loading = False

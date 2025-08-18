class SelectOutput:
    def __init__(self, gui: "PredictAppGui"):
        self.gui = gui

        # new_project
        self.new_project_name = Input(minlength=1, maxlength=255, placeholder="New Project Name")
        self.new_project_description = Text(
            "New project will be created. The created project will have the same dataset structure as the input project."
        )
        self.new_project_name_field = Field(
            content=self.new_project_name,
            title="New Project Name",
            description="Name of the new project to create for the results.",
        )

        self.append_description = Text("The results will be appended to the existing annotations.")
        self.replace_description = Text(
            "The existing annotations will be replaced with the predictions."
        )

        # iou_merge
        self.iou_merge_threshold = InputNumber(
            value=0, min=0, max=1, step=0.01, controls=False, width=200
        )
        self.iou_merge_description = Text(
            "If the prediction has IOU with any object greater than this value, it will be skipped."
        )
        self.iou_merge_threshold_field = Field(
            content=self.iou_merge_threshold,
            title="IOU Merge Threshold",
            description="Threshold for IOU merge. Float value between 0 and 1.",
        )

        self.radio = RadioGroup(
            items=[
                RadioGroup.Item(
                    "create",
                    "Create",
                    content=Container(
                        widgets=[self.new_project_description, self.new_project_name_field]
                    ),
                ),
                RadioGroup.Item(
                    "append",
                    "Append",
                    content=self.append_description,
                ),
                RadioGroup.Item(
                    "replace",
                    "Replace",
                    content=self.replace_description,
                ),
                RadioGroup.Item(
                    "iou_merge",
                    "IOU Merge",
                    content=Container(
                        widgets=[self.iou_merge_description, self.iou_merge_threshold_field]
                    ),
                ),
            ],
            direction="horizontal",
        )
        self.one_of = OneOf(self.radio)
        self.progress = Progress()
        self.validation_message = Text("", status="text")
        self.validation_message.hide()
        self.run_button = Button("Run", icon="zmdi zmdi-play")
        self.run_button.disable()
        self.result = ProjectThumbnail()
        self.result.hide()

        self.container = Container(
            widgets=[
                self.radio,
                self.one_of,
                self.validation_message,
                self.run_button,
                self.progress,
                self.result,
            ],
            direction="vertical",
            gap=20,
        )
        self.card = Card(title="Output", content=self.container)

    def set_result_thumbnail(self, project_id: int):
        try:
            project_info = self.gui.api.project.get_info_by_id(project_id)
            self.result.set(project_info)
            self.result.show()
        except Exception as e:
            logger.error(f"Failed to set result thumbnail: {str(e)}")
            self.result.hide()

    def get_output_settings(self):
        settings = {}
        mode = self.radio.get_value()
        if mode == "create":
            settings["mode"] = "create"
            settings["project_name"] = self.new_project_name.get_value()
        elif mode == "append":
            settings["mode"] = "append"
        elif mode == "replace":
            settings["mode"] = "replace"
        elif mode == "iou_merge":
            settings["mode"] = "iou_merge"
            settings["iou_merge_threshold"] = self.iou_merge_threshold.get_value()

        settings["mode"] = mode
        return settings

    def load_from_json(self, data):
        if not data:
            return
        mode = data["mode"]
        self.radio.set_value(mode)
        if mode == "create":
            self.new_project_name.set_value(data.get("project_name", ""))
        elif mode == "iou_merge":
            self.iou_merge_threshold.value = data.get("iou_merge_threshold", 0)

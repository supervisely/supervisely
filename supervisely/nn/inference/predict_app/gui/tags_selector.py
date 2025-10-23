from typing import Any, Dict

from supervisely.app.widgets import Button, Card, Container, TagsTable, Text


class TagsSelector:
    title = "Tags Selector"
    description = "Select tags that will be used for inference"
    lock_message = "Select previous step to unlock"

    def __init__(self):
        # Init Step
        self.display_widgets = []
        # -------------------------------- #

        # Init Base Widgets
        self.validator_text = None
        self.button = None
        self.container = None
        self.card = None
        # -------------------------------- #

        # Init Step Widgets
        self.tags_table = None
        # -------------------------------- #

        # Tags
        self.tags_table = TagsTable()
        self.tags_table.hide()
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.tags_table])
        # ----------------------------------- #

        # Base Widgets
        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        self.display_widgets.extend([self.validator_text, self.button])
        # -------------------------------- #

        # Card Layout
        self.container = Container(self.display_widgets)
        self.card = Card(
            title=self.title,
            description=self.description,
            content=self.container,
            lock_message=self.lock_message,
        )
        # -------------------------------- #

    @property
    def widgets_to_disable(self) -> list:
        return [self.tags_table]

    def load_from_json(self, data: Dict[str, Any]) -> None:
        if "tags" in data:
            self.set_tags(data["tags"])

    def get_selected_tags(self) -> list:
        return self.tags_table.get_selected_tags()

    def set_tags(self, tags) -> None:
        self.tags_table.select_tags(tags)

    def select_all_tags(self) -> None:
        self.tags_table.select_all()

    def get_settings(self) -> Dict[str, Any]:
        return {"tags": self.get_selected_tags()}

    def validate_step(self) -> bool:
        if self.tags_table.is_hidden():
            return True

        self.validator_text.hide()

        project_tags = self.tags_table.project_meta.tag_metas
        if len(project_tags) == 0:
            self.validator_text.set(text="Project has no tags", status="error")
            self.validator_text.show()
            return False

        selected_tags = self.tags_table.get_selected_tags()
        table_data = self.tags_table._table_data
        empty_tags = [
            row[0]["data"]
            for row in table_data
            if row[0]["data"] in selected_tags and row[2]["data"] == 0 and row[3]["data"] == 0
        ]

        n_tags = len(selected_tags)
        if n_tags == 0:
            message = "Please select at least one tag"
            status = "error"
        else:
            tag_text = "tag" if n_tags == 1 else "tags"
            message = f"Selected {n_tags} {tag_text}"
            status = "success"
            if empty_tags:
                intersections = set(selected_tags).intersection(empty_tags)
                if intersections:
                    tag_text = "tag" if len(intersections) == 1 else "tags"
                    message += (
                        f". Selected {tag_text} have no annotations: {', '.join(intersections)}"
                    )
                    status = "warning"

        self.validator_text.set(text=message, status=status)
        self.validator_text.show()
        return n_tags > 0

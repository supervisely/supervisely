from supervisely._utils import abs_url, is_debug_with_sly_net, is_development
from supervisely.app.widgets import Button, Card, Container, TagsTable, Text


class TagsSelector:
    title = "Tags Selector"
    description = "Select tags that will be used for training"
    lock_message = "Select previous step to unlock"

    def __init__(self, project_id: int, tags: list, app_options: dict = {}):
        # Init widgets
        self.qa_stats_text = None
        self.tags_table = None
        self.validator_text = None
        self.button = None
        self.container = None
        self.card = None
        # -------------------------------- #

        self.display_widgets = []
        self.app_options = app_options

        # GUI Components
        if is_development() or is_debug_with_sly_net():
            qa_stats_link = abs_url(f"projects/{project_id}/stats/datasets")
        else:
            qa_stats_link = f"/projects/{project_id}/stats/datasets"
        self.qa_stats_text = Text(
            text=f"<i class='zmdi zmdi-chart-donut' style='color: #7f858e'></i> <a href='{qa_stats_link}' target='_blank'> <b> QA & Stats </b></a>"
        )

        self.tags_table = TagsTable(project_id=project_id)
        if len(tags) > 0:
            self.tags_table.select_tags(tags)
        else:
            self.tags_table.select_all()

        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        self.display_widgets.extend(
            [self.qa_stats_text, self.tags_table, self.validator_text, self.button]
        )
        # -------------------------------- #

        self.container = Container(self.display_widgets)
        self.card = Card(
            title=self.title,
            description=self.description,
            content=self.container,
            lock_message=self.lock_message,
            collapsable=self.app_options.get("collapsable", False),
        )
        self.card.lock()

    @property
    def widgets_to_disable(self) -> list:
        return [self.tags_table]

    def get_selected_tags(self) -> list:
        return self.tags_table.get_selected_tags()

    def set_tags(self, tags) -> None:
        self.tags_table.select_tags(tags)

    def select_all_tags(self) -> None:
        self.tags_table.select_all()

    def validate_step(self) -> bool:
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

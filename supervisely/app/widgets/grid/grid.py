from typing import List
from supervisely.app.widgets import Widget, Container, generate_id
from supervisely.sly_logger import logger
from supervisely._utils import batched, rand_str


class Grid(Widget):
    def __init__(
        self,
        widgets: List[Widget],
        columns: int = 1,
        gap: int = 10,
        widget_id: str = None,
    ):
        self._widgets = widgets
        self._columns = columns
        self._gap = gap

        if self._columns < 1:
            raise ValueError(f"columns ({self._columns}) < 1")
        if self._columns > len(self._widgets):
            logger.warn(
                f"Number of columns ({self._columns}) > number of widgets ({len(self._widgets)}). Columns are set to {len(self._widgets)}"
            )
            self._columns = len(self._widgets)

        self._content = None
        if self._columns == 1:
            self._content = Container(
                direction="vertical",
                widgets=self._widgets,
                gap=self._gap,
                widget_id=generate_id(),
            )
        else:
            rows = []
            for batch in batched(self._widgets, batch_size=self._columns):
                rows.append(
                    Container(
                        direction="horizontal",
                        widgets=batch,
                        gap=self._gap,
                        widget_id=generate_id(),
                    )
                )
            self._content = Container(
                direction="vertical",
                widgets=rows,
                gap=self._gap,
                widget_id=generate_id(),
            )

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return None

    def get_json_state(self):
        return None

from supervisely.app.widgets import Widget


class LinePlot(Widget):
    def __init__(
        self,
        title: str,
        series: list = [],
        widget_id = None,
    ):
        self._title = title
        self._series = series
        self._options = {
            "title": self._title,
        }
        super(LinePlot, self).__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "title": self._title,
            "series": self._series,
            "options": self._options,
        }

    def get_json_state(self):
        return None

from supervisely.app.widgets import Widget
from supervisely.app.content import StateJson, DataJson


class LinePlot(Widget):
    def __init__(
        self,
        title: str,
        series: list = [],
        widget_id = None,
        yaxis_autorescale: bool = True,  # issue in apex, need to refresh page
    ):
        self._title = title
        self._series = series
        self._options = {
            "title": self._title,
            "yaxis": [{"show": True}],
        }
        self._yaxis_autorescale = yaxis_autorescale
        self._ymin = 0
        self._ymax = 10
        super(LinePlot, self).__init__(widget_id=widget_id, file_path=__file__)
        self.update_y_range(self._ymin, self._ymax)

    def get_json_data(self):
        return {
            "title": self._title,
            "series": self._series,
            "options": self._options,
            "ymin": self._ymin,
            "ymax": self._ymax,
        }

    def get_json_state(self):
        return None

    
    def update_y_range(self, ymin: int, ymax: int, send_changes=True):
        self._ymin = min(self._ymin, ymin)
        self._ymax = max(self._ymax, ymax)
        if self._yaxis_autorescale is False:
            self._options["yaxis"][0]["min"] = self._ymin
            self._options["yaxis"][0]["max"] = self._ymax

    def add_series(self, name: str, x: list, y: list, send_changes: bool = True):
        assert len(x) == len(y), ValueError(f"Lists x and y have different lenght, {len(x)} != {len(y)}")

        data = [{"x": px, "y": py} for px, py in zip(x, y)]
        series = {"name": name, "data": data}
        self._series.append(series)
        self.update_y_range(min(y), max(y))

        self.update_data()
        if send_changes:
            DataJson().send_changes()

    def add_series_batch(self, series: list):
        for serie in series:
            name = serie["name"]
            x = serie["x"]
            y = serie["y"]
            self.add_series(name, x, y, send_changes=False)
        DataJson().send_changes()

    def add_to_series(self, name_or_id: str or int, data: dict):
        if isinstance(name_or_id, int):
            series_id = name_or_id
        else:
            series_id, _ = self.get_series_by_name(name_or_id)
        self._series[series_id]['data'] +=  data
        DataJson().send_changes()

    def get_series_by_name(self, name):
        series_list = self.get_json_data()['series']
        series_id, series_data = next(((i, series) for i, series in enumerate(series_list) if series['name'] == name), (None, None))
        assert series_id is not None, KeyError("Series with name: {name} doesn't exists.")
        return series_id, series_data
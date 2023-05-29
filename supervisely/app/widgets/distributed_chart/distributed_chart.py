from typing import Union
from functools import wraps
from typing import NamedTuple, List
from supervisely.app.widgets.apexchart.apexchart import Apexchart
from supervisely.app.content import StateJson, DataJson

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class DistributedChart(Apexchart):
    class ClickedDataPoint(NamedTuple):
        series_index: int
        data_index: int
        data: dict
        
    def __init__(
        self,
        title: str,
        colors: List[str] = None,
    ):
        self._title = title
        self._series = []
        self._widget_height = 350
        
        if not colors:
            self._distributed = False
            self._colors = ["#008FFB"]
        else:
            self._distributed = True
            self._colors = colors
            
        self._options = {
            "chart": {
                "type": "treemap",
            },
            "legend": {
                "show": False,
            },
            "plotOptions": {
                "treemap": {
                    "distributed": self._distributed,
                    "enableShades": False,
                    }
                },
            "title": {"text": self._title, "align": "left"},
            }
        
        super(DistributedChart, self).__init__(
            series=self._series,
            options=self._options,
            type="treemap",
            height=self._widget_height,
        )
        
    def _manage_series(self, names: List[str], values: List[int], set: bool = False):
        if len(names) != len(values):
            raise ValueError(f"Names and values has different length: {len(names)} != {len(values)}")
        for value in values:
            if not isinstance(value, int):
                raise ValueError(f"All values must be ints, but {value} is {type(value)}.")
        
        data = [{"x": x, "y": y} for x, y in zip(names, values)]
        if set:
            self._series = [{"data": data}]
        else:
            self._series.append({"data": data})
        self.update_data()
        DataJson().send_changes()
        
    def add_series(self, names: List[str], values: List[int]):
        self._manage_series(names, values)
        
    def set_series(self, names: List[str], values: List[int]):
        self._manage_series(names, values, set=True)
        
    def get_series(self, index: int):
        try:
            return self._series[index]
        except IndexError:
            raise IndexError(f"Series with index {index} does not exist.")
        
    def delete_series(self, index: int):
        try:
            del self._series[index]
        except IndexError:
            raise IndexError(f"Series with index {index} does not exist.")
        self.update_data()
        DataJson().send_changes()
        
    def get_clicked_datapoint(self):
        value = self.get_clicked_value()
        series_index = value["seriesIndex"]
        data_index = value["dataPointIndex"]
        
        if series_index == -1 or data_index == -1:
            return
        
        raw_data = self._series[series_index]["data"][data_index]
        data = {
            "name": raw_data["x"],
            "value": raw_data["y"],
        }
        
        res = DistributedChart.ClickedDataPoint(series_index, data_index, data)
        
        return res
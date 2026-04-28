from typing import Union
from typing import NamedTuple, List, Dict
from supervisely.app.widgets.apexchart.apexchart import Apexchart
from supervisely.app.content import DataJson


class TreemapChart(Apexchart):
    """Widget for displaying data series by distribution. Supports multiple series and click events on datapoints."""

    class ClickedDataPoint(NamedTuple):
        """Class, representing clicked datapoint, which contains information about series, data index and data itself.
        It will be returned after click event on datapoint in immutable namedtuple
        with fields: series_index, data_index, data."""

        series_index: int
        data_index: int
        data: dict

    def __init__(
        self,
        title: str,
        colors: List[str] = None,
        tooltip: str = None,
    ):
        """Initialize the TreemapChart widget.

        :param title: Title of the chart displayed above.
        :type title: str
        :param colors: List of hex colors for series (e.g. #008FFB). Repeated if more series than colors.
        :type colors: List[str]
        :param tooltip: Tooltip displayed on hover over datapoint.
        :type tooltip: str

        :Usage Example:

            .. code-block:: python

                from supervisely.app.widgets import TreemapChart

                colors = ["#008FFB", "#00E396", "#FEB019", "#FF4560", "#775DD0"]
                tc = TreemapChart(title="Treemap Chart", colors=colors)
                names = ["cats", "dogs", "birds", "fishes", "snakes"]
                values = [3, 5, 1, 2, 1]
                tc.set_series(names, values)
        """
        self._title = title
        self._series = []
        self._widget_height = 350
        self._tooltip = tooltip

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
            "colors": self._colors,
        }

        sly_options = {}
        if self._tooltip is not None:
            sly_options["tooltip"] = self._tooltip
            self._options["tooltip"] = {"y": {}}

        super(TreemapChart, self).__init__(
            series=self._series,
            options=self._options,
            type="treemap",
            height=self._widget_height,
            sly_options=sly_options,
        )

    def _manage_series(self, names: List[str], values: List[Union[int, float]], set: bool = False):
        """This is a private method, which should not be used directly. Use add_series() or set_series() instead.
        It will add or set series to the chart, depending on set parameter. If set is True, all previous series will be
        deleted, otherwise new series will be added to the chart.

        :param names: list of names of the cells in series, which will be displayed on the chart
        :type names: List[str]
        :param values: list of values of the cells in series, which will be displayed on the chart
        :type values: List[Union[int, float]
        :param set: if True, all previous series will be deleted, otherwise new series will be added to the chart,
            defaults to False
        :type set: bool, optional
        :raises ValueError: if names and values has different length
        :raises ValueError: if any of values is not int or float
        """
        if len(names) != len(values):
            raise ValueError(
                f"Names and values has different length: {len(names)} != {len(values)}"
            )
        for value in values:
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"All values must be ints or floats, but {value} is {type(value)}."
                )

        data = [{"x": x, "y": y} for x, y in zip(names, values)]
        if set:
            self._series = [{"data": data}]
        else:
            self._series.append({"data": data})
        self.update_data()
        DataJson().send_changes()

    def add_series(self, names: List[str], values: List[Union[int, float]]):
        """Adds new series to the chart. Len of names and values must be equal, otherwise ValueError will be raised.

        :param names: list of names of the cells in series, which will be displayed on the chart
        :type names: List[str]
        :param values: list of values of the cells in series, which will be displayed on the chart
        :type values: List[Union[int, float]]
        """
        self._manage_series(names, values)

    def set_series(self, names: List[str], values: List[Union[int, float]]):
        """Sets series to the chart, deleting all previous series. Len of names and values must be equal,
        otherwise ValueError will be raised.

        :param names: list of names of the cells in series, which will be displayed on the chart
        :type names: List[str]
        :param values: list of values of the cells in series, which will be displayed on the chart
        :type values: List[Union[int, float]]
        """
        self._manage_series(names, values, set=True)

    def get_series(self, index: int) -> Dict[str, List[Dict[str, Union[int, float]]]]:
        """Returns series by index. If index is out of range, IndexError will be raised.
        Returned series is a dict with key "data" and a value is a list of dicts with keys "x" and "y",
        where "x" is a name of the cell and "y" is a value of the cell.

        :param index: index of the series, if index is out of range, IndexError will be raised
        :type index: int
        :raises TypeError: if index is not int
        :raises IndexError: if index is out of range
        :returns: series data by given index
        :rtype: Dict[str, Union[int, float]]
        """

        if not isinstance(index, int):
            raise TypeError(f"Index must be int, but {index} is {type(index)}.")
        try:
            return self._series[index]
        except IndexError:
            raise IndexError(f"Series with index {index} does not exist.")

    def delete_series(self, index: int):
        """Removes series by index from the chart. If index is out of range, IndexError will be raised.

        :param index: index of the series to delete
        :type index: int
        :raises TypeError: if index is not int
        :raises IndexError: if index is out of range
        """
        if not isinstance(index, int):
            raise TypeError(f"Index must be int, but {index} is {type(index)}.")
        try:
            del self._series[index]
        except IndexError:
            raise IndexError(f"Series with index {index} does not exist.")
        self.update_data()
        DataJson().send_changes()

    def get_clicked_datapoint(self) -> Union[ClickedDataPoint, None]:
        """Returns clicked datapoint as a ClickedDataPoint object, which is a namedtuple with fields:
        series_index, data_index and data. If click was outside of the cells, None will be returned.

        :returns: clicked datapoint as a ClickedDataPoint object or None if click was outside of the cells
        :rtype: Union[ClickedDataPoint, None]
        """
        value = self.get_clicked_value()
        series_index = value["seriesIndex"]
        data_index = value["dataPointIndex"]

        if series_index == -1 or data_index == -1:
            # If click was outside of the cells.
            return

        raw_data = self._series[series_index]["data"][data_index]
        data = {
            "name": raw_data["x"],
            "value": raw_data["y"],
        }

        res = TreemapChart.ClickedDataPoint(series_index, data_index, data)

        return res

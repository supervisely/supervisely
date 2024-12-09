from typing import Dict, List, Literal, Optional, Union

from supervisely.nn.benchmark.visualization.widgets.widget import BaseWidget


class ContainerWidget(BaseWidget):
    def __init__(
        self,
        widgets: List[BaseWidget],
        name: str = "container",
        title: str = None,
        grid: Optional[bool] = False,
        grid_cols: Union[int, Literal["auto"]] = 2,
        grid_rows: Union[int, Literal["auto"]] = "auto",
    ):
        super().__init__(name, title)
        self.widgets = widgets
        self.grid = grid
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows

    def to_html(self) -> str:
        if self.grid:
            if any([v != "auto" and type(v) != int for v in [self.grid_cols, self.grid_rows]]):
                raise ValueError("grid_cols and grid_rows must be either 'auto' or an integer")
            s = f"""
                <div 
                    id="{ self.id }"
                    style="
                        display: grid; 
                        grid-template-columns: repeat({self.grid_cols}, 1fr); 
                        grid-template-rows: repeat({self.grid_rows}, 1fr);
                        grid-column-gap: 16px;
                        grid-row-gap: 16px;
                        margin-bottom: 25px;
                    "
                >
            """
        else:
            s = "<div>"
        for widget in self.widgets:
            s += "<div>" + widget.to_html() + "</div>"
        s += "</div>"
        return s

    def save_data(self, basepath: str) -> None:
        for widget in self.widgets:
            widget.save_data(basepath)

    def get_state(self) -> Dict:
        state = {}
        for widget in self.widgets:
            state.update(widget.get_state())
        return state

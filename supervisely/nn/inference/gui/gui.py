from typing import List, Dict
from supervisely.app.widgets import (
    RadioTabs,
    RadioTable,
    Container,
    Field,
    Input,
    SelectString,
    Button,
)


def get_models_table_gui(models: List[Dict[str, str]]):
    # TODO: custom models are optional?

    cols = list(models[0].keys())
    rows = [list(model.values()) for model in models]
    models_table = RadioTable(cols, rows)
    models_table_field = Field(
        models_table, title="Pretrained Models", description="Choose model to serve"
    )
    device_select = SelectString(["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"])
    device_field = Field(device_select, title="Device")
    serve_btn = Button("SERVE")
    return Container([models_table_field, device_field, serve_btn])

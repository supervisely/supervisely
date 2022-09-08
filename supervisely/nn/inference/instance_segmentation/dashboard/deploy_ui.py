from supervisely.app.widgets import Text, Select, Field, Container, Button
from supervisely.imaging.color import hex2rgb

device = Select(
    items=[
        Select.Item(label="CPU", value="cpu"),
        Select.Item(label="GPU 0", value="cuda:0"),
        Select.Item(label="GPU 1", value="cuda:1"),
        Select.Item(label="GPU 2", value="cuda:2"),
        Select.Item(label="GPU 3", value="cuda:3"),
    ],
    filterable=True,
)

deploy = Button("Deploy model on device", icon="zmdi zmdi-flash")

device_field = Field(
    content=device,
    title="Select device",
    description="Model will be loaded (deployed) on selected device: CPU or GPU",
    icon=Field.Icon(
        zmdi_class="zmdi zmdi-memory",
        # color_rgb=hex2rgb("#4977ff"),
        # bg_color_rgb=hex2rgb("#ddf2ff"),
        # color_rgb=hex2rgb("#2cd26e"),
        # bg_color_rgb=hex2rgb("#d8f8e7"),
        bg_color_rgb=hex2rgb("#2cd26e"),
    ),
)

# how to hide widget?
deploy_status = Text("Model has been successfully deployed", "success")

layout = Container([device_field, deploy, deploy_status])

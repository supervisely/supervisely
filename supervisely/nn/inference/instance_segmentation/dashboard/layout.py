from supervisely.app.widgets import Text, Select, Menu, Field, Container

l = Text(text="left part", status="success")

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

ttt = Text(text="some text", status="warning")
# sidebar = sly.app.widgets.Sidebar(left_pane=l, right_pane=item)

device_field = Field(
    content=device,
    title="Select device",
    description="Model will be loaded (deployed) on selected device: CPU or GPU",
    icon=Field.Icon(zmdi_class="zmdi zmdi-memory"),
)

fields = Container([device_field])

g1 = Menu.Group(
    "Model",
    [
        Menu.Item(title="Deployment / Run", content=fields, icon="zmdi zmdi-dns"),
        Menu.Item(title="Classes", content=l, icon="zmdi zmdi-shape"),
        Menu.Item(title="Monitoring", content=l, icon="zmdi zmdi-chart"),
    ],
)
g2 = Menu.Group(
    "Preview predictions",
    [
        Menu.Item(title="Image", content=ttt, icon="zmdi zmdi-image"),
        Menu.Item(title="Video", content=ttt, icon="zmdi zmdi-youtube-play"),
    ],
)
g3 = Menu.Group(
    "Inference",
    [
        Menu.Item(
            title="Apply to images project",
            content=ttt,
            icon="zmdi zmdi-collection-folder-image",
        ),
        Menu.Item(
            title="Apply to videos project",
            content=ttt,
            icon="zmdi zmdi-collection-video",
        ),
    ],
)
menu = Menu(groups=[g1, g2, g3])
# menu = sly.app.widgets.Menu(items=g1_items)

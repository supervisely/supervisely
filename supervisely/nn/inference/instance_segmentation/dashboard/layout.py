from supervisely.app.widgets import Text, Select, Menu, Field, Container

l = Text(text="left part", status="success")
items = [
    Select.Item(label="CPU", value="cpu"),
    Select.Item(label="GPU 0", value="cuda:0"),
    Select.Item(value="option3"),
]
r = Select(items=items, filterable=True, placeholder="select me")

ttt = Text(text="some text", status="warning")
# sidebar = sly.app.widgets.Sidebar(left_pane=l, right_pane=item)

f0 = Field(r, "title0")
f1 = Field(r, "title1", "description1")
f2 = Field(r, "title2", "description2", title_url="/a/b")
f3 = Field(r, "title3", "description3", description_url="/a/b")
f4 = Field(r, "title4", "description4", title_url="/a/b", description_url="/a/b")
f5 = Field(r, "title5", "with icon", icon=Field.Icon(zmdi_class="zmdi zmdi-bike"))
f6 = Field(
    r,
    "title6",
    "with image",
    icon=Field.Icon(image_url="https://i.imgur.com/0E8d8bB.png"),
)

fields = Container([f0, f1, f2, f3, f4, f5, f6])

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

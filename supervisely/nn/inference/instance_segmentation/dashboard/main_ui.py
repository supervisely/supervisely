from supervisely.app.widgets import Text, Select, Menu, Field, Container, Button
from supervisely.imaging.color import hex2rgb
import supervisely.nn.inference.instance_segmentation.dashboard.deploy_ui as deploy_ui

l = Text(text="left part", status="success")


ttt = Text(text="some text", status="warning")
# sidebar = sly.app.widgets.Sidebar(left_pane=l, right_pane=item)

g1 = Menu.Group(
    "Model",
    [
        Menu.Item(
            title="Deployment / Run", content=deploy_ui.layout, icon="zmdi zmdi-dns"
        ),
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

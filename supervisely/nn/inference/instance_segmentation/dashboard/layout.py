from supervisely.app.widgets import Text, Select, Menu

l = Text(text="left part", status="success")
items = [
    Select.Item(label="CPU", value="cpu"),
    Select.Item(label="GPU 0", value="cuda:0"),
    Select.Item(value="option3"),
]
r = Select(items=items, filterable=True, placeholder="select me")

ttt = Text(text="some text", status="warning")
# sidebar = sly.app.widgets.Sidebar(left_pane=l, right_pane=item)

g1_items = [
    Menu.Item(title="Status", content=r, icon="zmdi zmdi-info"),
    Menu.Item(title="Classes", content=l, icon="zmdi zmdi-shape"),
    Menu.Item(title="Monitoring", content=l, icon="zmdi zmdi-chart"),
]
g2_items = [
    Menu.Item(title="m3", content=ttt),
    Menu.Item(title="m4"),
]
g1 = Menu.Group("Model", g1_items)
g2 = Menu.Group("Preview prediction", g2_items)
menu = Menu(groups=[g1, g2])
# menu = sly.app.widgets.Menu(items=g1_items)

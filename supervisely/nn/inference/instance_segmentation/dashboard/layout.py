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
    Menu.Item(title="m1", content=r),
    Menu.Item(title="m2", content=l),
]
g2_items = [
    Menu.Item(title="m3", content=ttt),
    Menu.Item(title="m4"),
]
g1 = Menu.Group("g1", g1_items)
g2 = Menu.Group("g2", g2_items)
menu = Menu(groups=[g1, g2])
# menu = sly.app.widgets.Menu(items=g1_items)

from supervisely.app.widgets import Container, OneOf, Text, Flexbox
import supervisely.nn.inference.instance_segmentation.dashboard.preview as preview


oneof_block = OneOf(preview.image_source)

t1 = Text("1")
t2 = Text("2")
t3 = Text("3")
t4 = Text("4")
t5 = Text("5")

flex_container = Flexbox([t1, t2, t3, t4, t5])

preview_image_layout = Container(
    [preview.image_source_field, oneof_block, flex_container],
)

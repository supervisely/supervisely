from supervisely.app.widgets import Text, Select, Field, OneOf


t1 = Text("url")
t2 = Text("sly")
t3 = Text("upload")

image_source = Select(
    items=[
        Select.Item(value="url", label="Image URL", content=t1),
        # Select.Item(value="demo", label="Demo image"), ???
        Select.Item(value="sly", label="Image in Supervisely", content=t2),
        Select.Item(value="upload", label="Upload your image", content=t3),
    ],
)

image_source_field = Field(
    image_source,
    "Select image",
    "Choose one of the image sources and provide test image",
)

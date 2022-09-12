from supervisely.app.widgets import Text, Select, Field, OneOf


t_url = Text("url")
t_sly = Text("sly")
t_upload = Text("upload")

image_source = Select(
    items=[
        Select.Item(value="url", label="Image URL", content=t_url),
        # Select.Item(value="demo", label="Demo image"), ???
        Select.Item(value="sly", label="Image in Supervisely", content=t_sly),
        Select.Item(value="upload", label="Upload your image", content=t_upload),
    ],
)

image_source_field = Field(
    image_source,
    "Select image",
    "Choose one of the image sources and provide test image",
)

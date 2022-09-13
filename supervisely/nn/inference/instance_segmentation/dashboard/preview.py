from supervisely.app.widgets import (
    Text,
    Select,
    Field,
    OneOf,
    SelectTeam,
    SelectWorkspace,
)


t_url = Text("url")
t_sly = Text("sly")
t_upload = Text("upload")

team_selector = SelectTeam()
workspace_selector = SelectWorkspace()

# select image
# select dataset
# select dataset
# Team->Workspace->Project->Dataset->Image
# - list images in dataset limit
# - dialog window?


image_source = Select(
    items=[
        Select.Item(value="url", label="Image URL", content=workspace_selector),
        Select.Item(value="sly", label="Image in Supervisely", content=t_sly),
        # Select.Item(value="demo", label="Demo image"), ???
        Select.Item(value="upload", label="Upload your image", content=t_upload),
    ],
)

image_source_field = Field(
    image_source,
    "Select image",
    "Choose one of the image sources and provide test image",
)

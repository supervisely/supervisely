from supervisely.app.widgets import (
    Text,
    Select,
    Field,
    OneOf,
    SelectTeam,
    SelectWorkspace,
    SelectProject,
)
from supervisely.app.widgets.select_dataset.select_dataset import SelectDataset
from supervisely.project.project_type import ProjectType

t_url = Text("url")
t_sly = Text("sly")
t_upload = Text("upload")

# selector = SelectTeam()
# selector = SelectWorkspace()
# selector = SelectProject(compact=False, project_types=[ProjectType.VIDEOS])
selector = SelectDataset(compact=False, multiselect=True)


# select image
# select dataset
# select dataset
# Team->Workspace->Project->Dataset->Image
# - list images in dataset limit
# - dialog window?


image_source = Select(
    items=[
        Select.Item(value="url", label="Image URL", content=selector),
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

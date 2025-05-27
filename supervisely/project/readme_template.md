This project uses Supervisely JSON format. Learn more about the format and the project in the README above.<br>

# General information

This is a general information about the project.<br>
{{general_info}}<br>

## Dataset structure

In this section, you can find information about the dataset structure. Dataset names are clickable and will redirect you to the corresponding folder.<br><br>
{{dataset_structure_info}}<br>

## Additional information (descriptions and custom data for datasets)

{{dataset_description_info}}

## Useful links

Please, visit the [Supervisely blog](https://supervisely.com/blog) to keep up with the latest news, updates, and tutorials and subscribe to [our YouTube channel](https://www.youtube.com/c/Supervisely) to watch video tutorials.<br>

-   [Supervisely Developer Portal](https://developer.supervisely.com/)
-   [Supervisely JSON format](https://docs.supervisely.com/data-organization/00_ann_format_navi)
-   [Intro to Python SDK](https://developer.supervisely.com/getting-started/intro-to-python-sdk)
-   [Python SDK reference](https://supervisely.readthedocs.io/en/latest/sdk_packages.html)
-   [API documentnation](https://api.docs.supervisely.com/)
-   [Supervisely Applications development](https://developer.supervisely.com/app-development/basics)
-   [Supervisely Ecosystem](https://ecosystem.supervisely.com/)

## Supervisely JSON format

In this section, you'll find a short description of the format and examples of working with it. To learn more about the format, please refer to the [Useful links](#useful-links) section.<br>

### Overview

Supervisely Annotation Format contains all the necessary information about the project, dataset, images, annotations, and tags in JSON format.<br>
A short overview of the format and its main features:

**Strictly defined data type** - only one type of data can be stored in one project, for example, images or videos, but not both. This is done to simplify the format and make it more universal.

**Project structure** - the project is divided into datasets, where each dataset contains corresponding data: images, videos, point clouds, etc and their annotations. Each dataset can contain any number of items and the project can contain any number of datasets. The datasets can also include other datasets, which allows you to create a hierarchical structure.

**Project Meta file** - the `meta.json` file contains information about object classes, tags, and other project settings. The `meta.json` file is located in the root directory of the project and is required for the correct operation of the format.

**Entities folder** - depending on the type of data stored in the project, dataset folders will contain one of the folders: `img`, `video`, `pointcloud`, `volume`. Each of these folders contains all the data of the corresponding type. For example, the `img` folder contains all images for the dataset.

**Annotations folder** - the `ann` folder contains all annotations for the dataset. Each annotation is stored in a separate file with the same name as the corresponding item. For example, the annotation for the image `img_001.jpg` will be stored in the file `img_001.jpg.json`.

ℹ️ To work with data in Supervisely JSON format, please install the latest Python SDK:

```bash
pip install supervisely
```

### Working with projects

Downloading the project from the platform to the local machine:

```python
import supervisely as sly

...

# If we don't know the project ID, we can get it by name:
project_id = api.project.get_info_by_name("my_project").id

# Download the project to the specified directory:
sly.Project.download(api, project_id, "/path/to/project")
```

Reading the local project:

```python
import supervisely as sly

...

project = sly.Project("/path/to/project", sly.OpenMode.READ)

# Do something with the project...

sly.Project.upload("/path/to/project", api, workspace_id)
```

### Project Meta

Project Meta is a crucial part of any project in Supervisely. It contains the essential information about the project - Classes and Tags. These are defined project-wide and can be used for labeling every dataset inside the current project. Project Meta is stored in the `meta.json` file in the root of the project folder. Each tag or class must be present in the project meta to be used in the project.

This is how we can download the project meta of an existing project:

```python
import supervisely as sly

api: sly.Api = sly.Api(server_address, token)
project_id = 123

# Important note: get_meta method returns a dictionary, not a ProjectMeta object!
meta_json = api.project.get_meta(project_id)

# We strongly recommend to always work with ProjectMeta objects, not with it's JSON representation!
# To convert JSON to ProjectMeta object, use ProjectMeta.from_json method
meta = sly.ProjectMeta.from_json(meta_json)

# So to do it in one line:
meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
```

It's a common use case to iterate over classes and tags in the project meta and it can be done easily with Supervisely Python SDK. Here's how you can do it:

```python
import supervisely as sly

# Let's imagine that we've already downloaded the project meta and now we want to iterate over classes and tags.
meta: sly.ProjectMeta

# Iterating over classes.
for obj_class in meta.obj_classes:
    # Each obj_class is an instance of sly.ObjClass, with a lot of useful fields and methods.
    # Check out [this section]() in the SDK Reference to learn more about ObjClass.
    print(obj_class.name)

# We can iterate over tags in a similar way.
for tag_meta in meta.tag_metas:
    # Each tag_meta is an instance of sly.TagMeta, it also has a lot of useful fields and methods.
    # Check out [this section]() in the SDK Reference to learn more about TagMeta.
    print(tag_meta.name)
```

### Working with Tags or Classes

In Supervisely tags provide an option to associate some additional information with the labeled image or the labels on it. Each tag can be attached to a single image or a single annotation only once, but there's no limit on how many times the same tag can be attached to different parts of the scene. There are different lists of tags for images and figures in the annotation file.
Classes are used to define the types of objects that can be labeled in the project. Each class has a name, a color, and a set of properties.

Creating a new tag and adding it to the project meta:

```python
import supervisely as sly

# Create an API client using environment variables.
api = sly.Api.from_env()

project_id = 123

# Retrieve project meta from the server.
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

# Create a new tag meta, using tag name and value type.
tag_meta = sly.TagMeta('like', sly.TagValueType.NONE)

# Add the new tag meta to the project meta.
new_project_meta = project_meta.add_tag_meta(tag_meta)

# Now, we can update the project meta on the server.
api.project.update_meta(project_id, new_project_meta)
```

### Working with Annotations

Retrieving image annotation from the platform:

```python
import supervisely as sly

api = sly.Api.from_env()

# Download image annotatation in JSON format.
image_id = 123
ann_json = api.annotation.download_json(image_id)

# We recommend to work with sly.Annotation object, not with raw JSON.
# To retrieve sly.Annotation object from JSON, you'll need the sly.ProjectMeta.

project_id = 456
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

# Now you can create sly.Annotation object from JSON and work with it.
ann = sly.Annotation.from_json(ann_json, project_meta)
```

Add a label to the image annotation and update it on the platform:

```python
import supervisely as sly
import cv2

# When creating a new annotation, you need to know the image size.
image_size = (300, 600)

# If you don't know the image size, you can read the image and get its size.
image_path = "path/to/image.jpg"
image_np = cv2.imread(image_path)
image_size = (image_np.shape[1], image_np.shape[0])

# Now, when you know the image size, you can create a new empty annotation.
ann = sly.Annotation(image_size)

# Let's now create a label, but of couese first we need to create a new object class.
class_kiwi = sly.ObjClass('kiwi', sly.Rectangle)

# So, we created a new Rectangle object class with the name 'kiwi'.
# Now, let's create a new label with this class and add it to the annotation.
label_kiwi = sly.Label(sly.Rectangle(0, 0, 300, 300), class_kiwi)

# Annotation object is immutable, so we'll need to create a new one, using
# add_label() method:
new_ann = ann.add_label(label_kiwi)

# Our annotation is ready, but before upload we must add this new object class to the ProjectMeta.
new_project_meta = project_meta.add_obj_class(class_kiwi)

# And now, we need to update the project meta on the server.
api.project.update_meta(project_id, new_project_meta)

# And finally, we can upload our new annotation to the server.
api.annotation.upload_ann(image_id, new_ann)
```

## Support and Feedback

If you have any questions or need assistance, please contact us in our [Community Slack](https://supervisely.slack.com/) or via [email](mailto:support@supervisely.com). We are always happy to help!

API References
==============

.. toctree::

   Public REST API              <https://api.docs.supervise.ly/>


Annotation
----------
**Annnotation**
- Working with labeling data of individual images. Annotation is the class that wraps all the labeling data for a given image: its Labels (geometrical objects) and Tags.

.. currentmodule:: supervisely.annotation

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    annotation
    label
    obj_class
    obj_class_collection
    tag
    tag_collection
    tag_meta
    tag_meta_collection

API
---
**API**
- Python wrappers to script your interactions with the Supervisely web instance. Instead of clicking around, you can write a script to request, via the API, a sequence of tasks, like training up a neural network and then running inference on a validation dataset.

.. currentmodule:: supervisely.api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    api
    agent_api
    annotation_api
    app_api
    dataset_api
    file_api.
    image_annotation_tool_api
    image_api
    import_storage_api
    labeling_job_api
    module_api
    neural_network_api
    object_class_api
    plugin_api
    project_api
    project_class_api
    role_api
    task_api
    team_api
    user_api
    workspace_api


Augmentation
------------
**Augmentation**
- Data augmentations to create more data variety for neural networks training.

.. currentmodule:: supervisely.aug

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    aug

Collection
----------

.. currentmodule:: supervisely.collection

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    key_indexed_collection

Decorators
----------

.. currentmodule:: supervisely.decorators

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    profile

Geometry
--------
**Geometry**
- All the logic concerned with working with geometric objects - compute statistics like object area, transform (rotate, scale, shift), extract bounding boxes, compute intersections and more.

.. currentmodule:: supervisely.geometry

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    any_geometry
    bitmap
    bitmap_base
    cuboid
    point
    point_location
    polygon
    polyline
    rectangle
    rotator
    vector_geometry
    graph

Imaging
-------
**Imaging**
- Our wrappers for working with images. IO, transformations, text rendering, color conversions.

.. currentmodule:: supervisely.imaging

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    font
    color
    image

IO
--
**IO**
- Low-level convenience IO wrappers that we found useful internally.

.. currentmodule:: supervisely.io

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    fs
    json

Labeling Jobs
-------------

.. currentmodule:: supervisely.labeling_jobs

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    utils


Pointcloud
----------

.. currentmodule:: supervisely.pointcloud

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    pointcloud

Pointcloud Annotation
---------------------

.. currentmodule:: supervisely.pointcloud_annotation

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-module-template.rst

   pointcloud_annotation
   pointcloud_figure
   pointcloud_object
   pointcloud_object_Collection

Project
-------
**Project**
- Working with Superrvisely projects on disk.

.. currentmodule:: supervisely.project

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-module-template.rst

   project
   pointcloud-project
   project_meta
   project_type
   video_project

Task
----
**Task**
- Constants defining the directories where a plugin should expect the input and output data to be. Also helpers to stream progress data from a running plugin back to the web instance.

.. currentmodule:: supervisely.task

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-module-template.rst

   progress

User
----

.. currentmodule:: supervisely.user

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-module-template.rst

   user

Video
-----

.. currentmodule:: supervisely.video

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-module-template.rst

   video

Video Annotation
----------------

.. currentmodule:: supervisely.video_annotation

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-module-template.rst

   frame
   frame_collection
   key_id_ap
   video_annotation
   video_figure
   video_object
   video_object_collection
   video_tag
   video_tag_collection
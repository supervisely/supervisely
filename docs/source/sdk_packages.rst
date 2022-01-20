API References
==============

Annotation
----------
**Annnotation**
- Working with labeling data of individual images. Annotation is the class that wraps all the labeling data for a given image: its Labels (geometrical objects) and Tags.

.. currentmodule:: supervisely.annotation

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    supervisely.annotation.annotation.Annotation
    supervisely.annotation.label.Label
    supervisely.annotation.label.LabelBase
    supervisely.annotation.obj_class.ObjClass
    supervisely.annotation.obj_class_collection.ObjClassCollection
    supervisely.annotation.tag.Tag
    supervisely.annotation.tag_collection.TagCollection
    supervisely.annotation.tag_meta.TagMeta
    supervisely.annotation.tag_meta_collection.TagMetaCollection

API
---
**API**
- Python wrappers to script your interactions with the Supervisely web instance. Instead of clicking around, you can write a script to request, via the API, a sequence of tasks, like training up a neural network and then running inference on a validation dataset.

.. currentmodule:: supervisely.api

.. autosummary::
    :toctree: api
    :nosignatures:

    supervisely.api.api.API
    supervisely.api.agent_api.AgentApi
    supervisely.api.annotation_api.AnnotationApi
    supervisely.api.app_api.AppApi
    supervisely.api.dataset_api.DatasetApi
    supervisely.api.file_api.FileApi
    supervisely.api.annotation_tool_api.ImageAnnotationToolApi
    supervisely.api.image_api.ImageApi
    supervisely.api.import_storage_api.ImportStorageApi
    supervisely.api.labeling_job_api.LabelingJobApi
    supervisely.api.module_api.ModuleApi
    supervisely.api.neural_network_api.NeuralNetworkApi
    supervisely.api.object_class_api.ObjectClassApi
    supervisely.api.plugin_api.PluginApi
    supervisely.api.project_api.ProjectApi
    supervisely.api.project_class_api.ProjectClassApi
    supervisely.api.role_api.RoleApi
    supervisely.api.task_api.TaskApi
    supervisely.api.team_api.TeamApi
    supervisely.api.user_api.UserApi
    supervisely.api.workspace_api.WorkspaceApi
    supervisely.api.entity_annotation_api.EntityAnnotationApi
    supervisely.api.pointcloud_api.PointcloudApi
    supervisely.api.video_api.VideoApi
    sdcsdwer


Augmentation
------------
**Augmentation**
- Data augmentations to create more data variety for neural networks training.

.. currentmodule:: supervisely.aug.aug

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Augmentation

Collection
----------

.. currentmodule:: supervisely.collection.key_indexed_collection

.. autosummary::
    :toctree: api
    :nosignatures:

    KeyObject
    KeyIndexedCollection

Decorators
----------

.. currentmodule:: supervisely.decorators

.. autosummary::
    :toctree: api
    :nosignatures:

    Profile

Geometry
--------
**Geometry**
- All the logic concerned with working with geometric objects - compute statistics like object area, transform (rotate, scale, shift), extract bounding boxes, compute intersections and more.

.. currentmodule:: supervisely_lib.geometry

.. autosummary::
    :toctree: api
    :nosignatures:

   Any Geometry
   Bitmap
   Bitmap_base
   Cuboid
   Point
   Point_location
   Polygon
   Polyline
   Rectangle
   Rotator
   Vector Geometry
   Graph

Imaging
-------
**Imaging**
- Our wrappers for working with images. IO, transformations, text rendering, color conversions.

.. currentmodule:: supervisely_lib.imaging

.. autosummary::
    :toctree: api
    :nosignatures:

   Font
   Color
   Image

IO
--
**IO**
- Low-level convenience IO wrappers that we found useful internally.

.. currentmodule:: supervisely.io

.. autosummary::
    :toctree: api
    :nosignatures:

   Fs
   Json

Labeling Jobs
-------------

.. currentmodule:: supervisely.labeling_jobs

.. autosummary::
    :toctree: api
    :nosignatures:

   Utilities

Pointcloud
----------

.. currentmodule:: supervisely.pointcloud.pointcloud

.. autosummary::
    :toctree: api
    :nosignatures:

    Pointcloud

Pointcloud Annotation
---------------------

.. currentmodule:: supervisely.pointcloud_annotation

.. autosummary::
    :toctree: api
    :nosignatures:

   Constants
   Pointcloud Annotation
   Pointcloud Figure
   Pointcloud Object
   Pointcloud Object Collection

Project
-------
**Project**
- Working with Superrvisely projects on disk.

.. currentmodule:: supervisely.project

.. autosummary::
    :toctree: api
    :nosignatures:

   Project
   Pointcloud Project
   ProjectMeta
   ProjectType
   Video Project

Task
----
**Task**
- Constants defining the directories where a plugin should expect the input and output data to be. Also helpers to stream progress data from a running plugin back to the web instance.

.. currentmodule:: supervisely.task

.. autosummary::
    :toctree: api
    :nosignatures:

   Progress

User
----

.. currentmodule:: supervisely.user.user

.. autosummary::
    :toctree: api
    :nosignatures:

   User

Video
-----

.. currentmodule:: supervisely.video.video

.. autosummary::
    :toctree: api
    :nosignatures:

   Video

Video Annotation
----------------

.. currentmodule:: supervisely.video_annotation

.. autosummary::
    :toctree: api
    :nosignatures:

   Frame
   Frame Collection
   Key ID Map
   Video Annotation
   Video Figure
   Video Object
   Video Object Collection
   Video Tag
   Video Tag Collection
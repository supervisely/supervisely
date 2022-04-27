Python SDK
==========

Annotation
----------
**Annnotation**
- Working with labeling data of individual images. Annotation is the class that wraps all the labeling data for a given image: its Labels (geometrical objects) and Tags.

.. currentmodule:: supervisely.annotation.annotation

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Annotation
    AnnotationJsonFields


.. currentmodule:: supervisely.annotation.label

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Label
    LabelBase
    LabelJsonFields


.. currentmodule:: supervisely.annotation.obj_class

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ObjClass
    ObjClassJsonFields


.. currentmodule:: supervisely.annotation.obj_class_collection

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ObjClassCollection


.. currentmodule:: supervisely.annotation.tag

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Tag
    TagJsonFields


.. currentmodule:: supervisely.annotation.tag_collection

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    TagCollection


.. currentmodule:: supervisely.annotation.tag_meta

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    TagMeta
    TagMetaJsonFields
    TagApplicableTo
    TagValueType


.. currentmodule:: supervisely.annotation.tag_meta_collection

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    TagMetaCollection

API
---
**API**
- Python wrappers to script your interactions with the Supervisely web instance. Instead of clicking around, you can write a script to request, via the API, a sequence of tasks, like training up a neural network and then running inference on a validation dataset.

.. currentmodule:: supervisely.api.api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Api

.. currentmodule:: supervisely.api.agent_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    AgentApi

.. currentmodule:: supervisely.api.annotation_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    AnnotationApi

.. currentmodule:: supervisely.api.app_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    AppApi

.. currentmodule:: supervisely.api.dataset_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    DatasetApi

.. currentmodule:: supervisely.api.file_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    FileApi

.. currentmodule:: supervisely.api.image_annotation_tool_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ImageAnnotationToolApi
    ImageAnnotationToolAction

.. currentmodule:: supervisely.api.image_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ImageApi

.. currentmodule:: supervisely.api.import_storage_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ImportStorageApi

.. currentmodule:: supervisely.api.labeling_job_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    LabelingJobApi

.. currentmodule:: supervisely.api.module_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ApiField
    ModuleApi
    ModuleApiBase
    CloneableModuleApi
    UpdateableModule
    RemoveableModuleApi
    RemoveableBulkModuleApi
    ModuleWithStatus

.. currentmodule:: supervisely.api.neural_network_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    NeuralNetworkApi

.. currentmodule:: supervisely.api.object_class_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ObjectClassApi

.. currentmodule:: supervisely.api.plugin_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PluginApi

.. currentmodule:: supervisely.api.project_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ProjectApi

.. currentmodule:: supervisely.api.project_class_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ProjectClassApi

.. currentmodule:: supervisely.api.role_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    RoleApi

.. currentmodule:: supervisely.api.task_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    TaskApi

.. currentmodule:: supervisely.api.team_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ActivityAction
    TeamApi

.. currentmodule:: supervisely.api.user_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    UserApi

.. currentmodule:: supervisely.api.workspace_api

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    WorkspaceApi


Augmentation
------------
**Augmentation**
- Data augmentations to create more data variety for neural networks training.

.. currentmodule:: supervisely.aug

.. autosummary::
    :toctree: rst_templates/api
    :template: autosummary/custom-module-template.rst

    aug

Collection
----------

.. currentmodule:: supervisely.collection

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    key_indexed_collection

Decorators
----------

.. currentmodule:: supervisely.decorators

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    profile

Geometry
--------
**Geometry**
- All the logic concerned with working with geometric objects - compute statistics like object area, transform (rotate, scale, shift), extract bounding boxes, compute intersections and more.

.. currentmodule:: supervisely.geometry

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

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
    :template: autosummary/custom-class-template.rst

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
    :template: autosummary/custom-class-template.rst

    fs
    json

Labeling Jobs
-------------

.. currentmodule:: supervisely.labeling_jobs

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    utils


Pointcloud
----------

.. currentmodule:: supervisely.pointcloud

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    pointcloud

Pointcloud Annotation
---------------------

.. currentmodule:: supervisely.pointcloud_annotation

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

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
    :template: autosummary/custom-class-template.rst

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
    :template: autosummary/custom-class-template.rst

   progress

User
----

.. currentmodule:: supervisely.user

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

   user

Video
-----

.. currentmodule:: supervisely.video

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

   video

Video Annotation
----------------

.. currentmodule:: supervisely.video_annotation

.. autosummary::
    :toctree: rst_templates/api
    :nosignatures:
    :template: autosummary/custom-class-template.rst

   frame
   frame_collection
   key_id_ap
   video_annotation
   video_figure
   video_object
   video_object_collection
   video_tag
   video_tag_collection

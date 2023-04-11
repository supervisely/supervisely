SDK Reference
==============

Annotation
----------
**Annnotation**
- Working with labeling data of individual images. Annotation is the class that wraps all the labeling data for a given image: its Labels (geometrical objects) and Tags.

.. currentmodule:: supervisely.annotation.annotation

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Annotation
    AnnotationJsonFields

.. currentmodule:: supervisely.annotation.label

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Label
    LabelJsonFields

.. currentmodule:: supervisely.annotation.obj_class

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ObjClass
    ObjClassJsonFields

.. currentmodule:: supervisely.annotation.obj_class_collection

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ObjClassCollection

.. currentmodule:: supervisely.annotation.tag

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Tag
    TagJsonFields

.. currentmodule:: supervisely.annotation.tag_collection

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    TagCollection

.. currentmodule:: supervisely.annotation.tag_meta

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    TagMeta
    TagMetaJsonFields
    TagApplicableTo
    TagValueType

.. currentmodule:: supervisely.annotation.tag_meta_collection

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    TagMetaCollection

API
---
**API**
- Python wrappers to script your interactions with the Supervisely web instance. Instead of clicking around, you can write a script to request, via the API, a sequence of tasks, like training up a neural network and then running inference on a validation dataset.

.. currentmodule:: supervisely.api.api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Api

.. currentmodule:: supervisely.api.agent_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    AgentApi

.. currentmodule:: supervisely.api.annotation_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    AnnotationApi

.. currentmodule:: supervisely.api.app_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    AppApi

.. currentmodule:: supervisely.api.dataset_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    DatasetApi

.. currentmodule:: supervisely.api.file_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    FileApi

.. currentmodule:: supervisely.api.image_annotation_tool_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ImageAnnotationToolApi
    ImageAnnotationToolAction

.. currentmodule:: supervisely.api.image_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ImageApi
    ImageInfo

.. currentmodule:: supervisely.api.import_storage_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ImportStorageApi

.. currentmodule:: supervisely.api.labeling_job_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    LabelingJobApi

.. currentmodule:: supervisely.api.module_api

.. autosummary::
    :toctree: sdk
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
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    NeuralNetworkApi

.. currentmodule:: supervisely.api.object_class_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ObjectClassApi

.. currentmodule:: supervisely.api.plugin_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PluginApi

.. currentmodule:: supervisely.api.project_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ProjectApi

.. currentmodule:: supervisely.api.project_class_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ProjectClassApi

.. currentmodule:: supervisely.api.role_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    RoleApi

.. currentmodule:: supervisely.api.task_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    TaskApi

.. currentmodule:: supervisely.api.team_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ActivityAction
    TeamApi

.. currentmodule:: supervisely.api.user_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    UserApi

.. currentmodule:: supervisely.api.workspace_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    WorkspaceApi

Video API
------------
**Video API**
- API for working with videos in Supervisely.

.. currentmodule:: supervisely.api.video.video_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VideoApi
    VideoInfo

.. currentmodule:: supervisely.api.video.video_annotation_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VideoAnnotationAPI


.. currentmodule:: supervisely.api.video.video_figure_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VideoFigureApi

.. currentmodule:: supervisely.api.video.video_frame_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VideoFrameAPI

.. currentmodule:: supervisely.api.video.video_object_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VideoObjectApi

.. currentmodule:: supervisely.api.video.video_tag_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VideoTagApi

Volume API
------------
**Volume API**
- API for working with volumes in Supervisely.

.. currentmodule:: supervisely.api.volume.volume_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VolumeApi
    VolumeInfo

.. currentmodule:: supervisely.api.volume.volume_annotation_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VolumeAnnotationAPI

.. currentmodule:: supervisely.api.volume.volume_figure_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VolumeFigureApi

.. currentmodule:: supervisely.api.volume.volume_object_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VolumeObjectApi

.. currentmodule:: supervisely.api.volume.volume_tag_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VolumeTagApi

Pointcloud API
------------
**Pointcloud API**
- API for working with pointclouds in Supervisely.

.. currentmodule:: supervisely.api.pointcloud.pointcloud_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PointcloudApi
    PointcloudInfo

.. currentmodule:: supervisely.api.pointcloud.pointcloud_episode_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PointcloudEpisodeApi

.. currentmodule:: supervisely.api.pointcloud.pointcloud_annotation_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PointcloudAnnotationAPI

.. currentmodule:: supervisely.api.pointcloud.pointcloud_episode_annotation_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PointcloudEpisodeAnnotationAPI

.. currentmodule:: supervisely.api.pointcloud.pointcloud_figure_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PointcloudFigureApi

.. currentmodule:: supervisely.api.pointcloud.pointcloud_object_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PointcloudObjectApi

.. currentmodule:: supervisely.api.pointcloud.pointcloud_tag_api

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PointcloudTagApi

Augmentation
------------
**Augmentation**
- Data augmentations to create more data variety for neural networks training.

.. currentmodule:: supervisely.aug

.. autosummary::
    :toctree: sdk
    :template: autosummary/custom-module-template.rst

    aug

Collection
----------

.. currentmodule:: supervisely.collection.key_indexed_collection

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    DuplicateKeyError
    KeyObject
    KeyIndexedCollection
    MultiKeyIndexedCollection

Decorators
----------

.. currentmodule:: supervisely.decorators

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    profile

Geometry
--------
**Geometry**
- All the logic concerned with working with geometric objects - compute statistics like object area, transform (rotate, scale, shift), extract bounding boxes, compute intersections and more.

.. currentmodule:: supervisely.geometry.any_geometry

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    AnyGeometry

.. currentmodule:: supervisely.geometry.bitmap

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Bitmap
    SkeletonizeMethod

.. currentmodule:: supervisely.geometry.bitmap_base

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    BitmapBase

.. currentmodule:: supervisely.geometry.cuboid

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Cuboid
    CuboidFace

.. currentmodule:: supervisely.geometry.cuboid_3d

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Cuboid3d
    Vector3d

.. currentmodule:: supervisely.geometry.geometry

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Geometry

.. currentmodule:: supervisely.geometry.point

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Point

.. currentmodule:: supervisely.geometry.point_3d

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Point3d

.. currentmodule:: supervisely.geometry.point_location

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PointLocation

.. currentmodule:: supervisely.geometry.polygon

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Polygon

.. currentmodule:: supervisely.geometry.polyline

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Polyline

.. currentmodule:: supervisely.geometry.rectangle

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Rectangle

.. currentmodule:: supervisely.geometry.image_rotator

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ImageRotator

.. currentmodule:: supervisely.geometry.vector_geometry

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VectorGeometry

.. currentmodule:: supervisely.geometry.graph

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Node
    GraphNodes

Imaging
-------
**Imaging**
- Our wrappers for working with images. IO, transformations, text rendering, color conversions.

.. currentmodule:: supervisely.imaging

.. autosummary::
    :toctree: sdk
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
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    fs
    json


Labeling Jobs
-------------

.. currentmodule:: supervisely.labeling_jobs

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    utils


Pointcloud
----------

.. currentmodule:: supervisely.pointcloud

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    pointcloud

Pointcloud Annotation
---------------------

.. currentmodule:: supervisely.pointcloud_annotation.pointcloud_annotation

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PointcloudAnnotation

.. currentmodule:: supervisely.pointcloud_annotation.pointcloud_figure

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PointcloudFigure

.. currentmodule:: supervisely.pointcloud_annotation.pointcloud_object

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PointcloudObject

.. currentmodule:: supervisely.pointcloud_annotation.pointcloud_object_collection

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PointcloudObjectCollection

.. currentmodule:: supervisely.pointcloud_annotation.pointcloud_tag

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PointcloudTag

.. currentmodule:: supervisely.pointcloud_annotation.pointcloud_tag_collection

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PointcloudTagCollection

.. currentmodule:: supervisely.pointcloud_annotation.pointcloud_episode_annotation

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PointcloudEpisodeAnnotation

Project
-------
**Project**
- Working with Supervisely projects on disk.

.. currentmodule:: supervisely.project.project

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Project
    DatasetDict
    Dataset
    OpenMode
    ItemPaths
    ItemInfo

.. currentmodule:: supervisely.project.pointcloud_project

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    PointcloudProject
    PointcloudDataset

.. currentmodule:: supervisely.project.project_meta

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ProjectMeta
    ProjectMetaJsonFields

.. currentmodule:: supervisely.project.project_type

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ProjectType

.. currentmodule:: supervisely.project.video_project

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VideoProject
    VideoDataset
    VideoItemPaths

.. currentmodule:: supervisely.project.volume_project

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VolumeProject
    VolumeDataset

Task
----
**Task**
- Constants defining the directories where a plugin should expect the input and output data to be. Also helpers to stream progress data from a running plugin back to the web instance.

.. currentmodule:: supervisely.task.progress

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Progress

User
----

.. currentmodule:: supervisely.user.user

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    UserRoleName

Video
-----

.. currentmodule:: supervisely.video

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    video

Video Annotation
----------------

.. currentmodule:: supervisely.video_annotation.video_annotation

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VideoAnnotation

.. currentmodule:: supervisely.video_annotation.frame

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    Frame

.. currentmodule:: supervisely.video_annotation.frame_collection

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    FrameCollection

.. currentmodule:: supervisely.video_annotation.key_id_map

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    KeyIdMap

.. currentmodule:: supervisely.video_annotation.video_figure

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VideoFigure

.. currentmodule:: supervisely.video_annotation.video_object

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VideoObject

.. currentmodule:: supervisely.video_annotation.video_object_collection

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

      VideoObjectCollection

.. currentmodule:: supervisely.video_annotation.video_tag

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    VideoTag

.. currentmodule:: supervisely.video_annotation.video_tag_collection

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

      VideoTagCollection

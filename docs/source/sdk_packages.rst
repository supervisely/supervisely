SDK Reference
==============

Annotation
----------
Working with labeling data of individual images. Annotation wraps all the labeling data for a given image: its Labels (geometrical objects) and Tags.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ~supervisely.annotation.annotation.Annotation
    ~supervisely.annotation.annotation.AnnotationJsonFields
    ~supervisely.annotation.label.Label
    ~supervisely.annotation.label.LabelJsonFields
    ~supervisely.annotation.obj_class.ObjClass
    ~supervisely.annotation.obj_class.ObjClassJsonFields
    ~supervisely.annotation.obj_class_collection.ObjClassCollection
    ~supervisely.annotation.tag.Tag
    ~supervisely.annotation.tag.TagJsonFields
    ~supervisely.annotation.tag_collection.TagCollection
    ~supervisely.annotation.tag_meta.TagMeta
    ~supervisely.annotation.tag_meta.TagMetaJsonFields
    ~supervisely.annotation.tag_meta.TagApplicableTo
    ~supervisely.annotation.tag_meta.TagValueType
    ~supervisely.annotation.tag_meta_collection.TagMetaCollection

API
---
Python wrappers to script your interactions with the Supervisely web instance. Instead of clicking around, you can write a script to request, via the API, a sequence of tasks (e.g. training a neural network and then running inference on a validation dataset).

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ~supervisely.api.api.Api
    ~supervisely.api.agent_api.AgentApi
    ~supervisely.api.annotation_api.AnnotationApi
    ~supervisely.api.app_api.AppApi
    ~supervisely.api.app_api.WorkflowSettings
    ~supervisely.api.app_api.WorkflowMeta
    ~supervisely.api.dataset_api.DatasetApi
    ~supervisely.api.file_api.FileApi
    ~supervisely.api.guides_api.GuidesApi
    ~supervisely.api.storage_api.StorageApi
    ~supervisely.api.image_annotation_tool_api.ImageAnnotationToolApi
    ~supervisely.api.image_annotation_tool_api.ImageAnnotationToolAction
    ~supervisely.api.image_api.ImageApi
    ~supervisely.api.image_api.ImageInfo
    ~supervisely.api.image_api.BlobImageInfo
    ~supervisely.api.import_storage_api.ImportStorageApi
    ~supervisely.api.issues_api.IssuesApi
    ~supervisely.api.labeling_job_api.LabelingJobApi
    ~supervisely.api.entities_collection_api.EntitiesCollectionApi
    ~supervisely.api.labeling_queue_api.LabelingQueueApi
    ~supervisely.api.module_api.ApiField
    ~supervisely.api.module_api.ModuleApi
    ~supervisely.api.module_api.ModuleApiBase
    ~supervisely.api.module_api.CloneableModuleApi
    ~supervisely.api.module_api.UpdateableModule
    ~supervisely.api.module_api.RemoveableModuleApi
    ~supervisely.api.module_api.RemoveableBulkModuleApi
    ~supervisely.api.module_api.ModuleWithStatus
    ~supervisely.api.object_class_api.ObjectClassApi
    ~supervisely.api.plugin_api.PluginApi
    ~supervisely.api.project_api.ProjectApi
    ~supervisely.api.project_class_api.ProjectClassApi
    ~supervisely.api.role_api.RoleApi
    ~supervisely.api.task_api.TaskApi
    ~supervisely.api.team_api.ActivityAction
    ~supervisely.api.team_api.TeamApi
    ~supervisely.api.user_api.UserApi
    ~supervisely.api.workspace_api.WorkspaceApi


Video API
---------
API for working with videos in Supervisely.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ~supervisely.api.video.video_api.VideoApi
    ~supervisely.api.video.video_api.VideoInfo
    ~supervisely.api.video.video_annotation_api.VideoAnnotationAPI
    ~supervisely.api.video.video_figure_api.VideoFigureApi
    ~supervisely.api.video.video_frame_api.VideoFrameAPI
    ~supervisely.api.video.video_object_api.VideoObjectApi
    ~supervisely.api.video.video_tag_api.VideoTagApi

Volume API
----------
API for working with volumes in Supervisely.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ~supervisely.api.volume.volume_api.VolumeApi
    ~supervisely.api.volume.volume_api.VolumeInfo
    ~supervisely.api.volume.volume_annotation_api.VolumeAnnotationAPI
    ~supervisely.api.volume.volume_figure_api.VolumeFigureApi
    ~supervisely.api.volume.volume_object_api.VolumeObjectApi
    ~supervisely.api.volume.volume_tag_api.VolumeTagApi

Pointcloud API
--------------
API for working with pointclouds in Supervisely.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ~supervisely.api.pointcloud.pointcloud_api.PointcloudApi
    ~supervisely.api.pointcloud.pointcloud_api.PointcloudInfo
    ~supervisely.api.pointcloud.pointcloud_episode_api.PointcloudEpisodeApi
    ~supervisely.api.pointcloud.pointcloud_annotation_api.PointcloudAnnotationAPI
    ~supervisely.api.pointcloud.pointcloud_episode_annotation_api.PointcloudEpisodeAnnotationAPI
    ~supervisely.api.pointcloud.pointcloud_figure_api.PointcloudFigureApi
    ~supervisely.api.pointcloud.pointcloud_object_api.PointcloudObjectApi
    ~supervisely.api.pointcloud.pointcloud_tag_api.PointcloudTagApi

Neural Networks API
-------------------
APIs for deploying models, running inference, and working with model metadata in Supervisely.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ~supervisely.api.nn.deploy_api.DeployApi
    ~supervisely.api.nn.neural_network_api.NeuralNetworkApi
    ~supervisely.api.nn.train_api.TrainApi
    ~supervisely.nn.model.model_api.ModelAPI
    ~supervisely.nn.model.prediction.Prediction
    ~supervisely.nn.model.prediction_session.PredictionSession

Augmentation
------------
Data augmentations to create more data variety for neural networks training.

.. autosummary::
    :toctree: sdk
    :template: autosummary/custom-module-template.rst

    ~supervisely.aug.aug

Collection
----------
Key-indexed collections and helpers used across the SDK.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ~supervisely.collection.key_indexed_collection.DuplicateKeyError
    ~supervisely.collection.key_indexed_collection.KeyObject
    ~supervisely.collection.key_indexed_collection.KeyIndexedCollection
    ~supervisely.collection.key_indexed_collection.MultiKeyIndexedCollection

Decorators
----------
Small utility decorators (e.g. profiling helpers).

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    ~supervisely.decorators.profile

Geometry
--------
All the logic concerned with working with geometric objects - compute statistics like object area, transform (rotate, scale, shift), extract bounding boxes, compute intersections and more.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ~supervisely.geometry.any_geometry.AnyGeometry
    ~supervisely.geometry.bitmap.Bitmap
    ~supervisely.geometry.bitmap.SkeletonizeMethod
    ~supervisely.geometry.alpha_mask.AlphaMask
    ~supervisely.geometry.bitmap_base.BitmapBase
    ~supervisely.geometry.cuboid.Cuboid
    ~supervisely.geometry.cuboid.CuboidFace
    ~supervisely.geometry.cuboid_3d.Cuboid3d
    ~supervisely.geometry.cuboid_3d.Vector3d
    ~supervisely.geometry.mask_3d.Mask3D
    ~supervisely.geometry.geometry.Geometry
    ~supervisely.geometry.point.Point
    ~supervisely.geometry.point_3d.Point3d
    ~supervisely.geometry.point_location.PointLocation
    ~supervisely.geometry.polygon.Polygon
    ~supervisely.geometry.polyline.Polyline
    ~supervisely.geometry.rectangle.Rectangle
    ~supervisely.geometry.image_rotator.ImageRotator
    ~supervisely.geometry.vector_geometry.VectorGeometry
    ~supervisely.geometry.graph.Node
    ~supervisely.geometry.graph.GraphNodes

Imaging
-------
Wrappers for working with images: IO, transformations, text rendering, color conversions.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    ~supervisely.imaging.font
    ~supervisely.imaging.color
    ~supervisely.imaging.image

IO
--
Low-level convenience IO wrappers that are used across the SDK.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    ~supervisely.io.fs
    ~supervisely.io.json


Labeling Jobs
-------------
Helpers for working with labeling jobs and related utilities.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    ~supervisely.labeling_jobs.utils


Pointcloud
----------
Core pointcloud-related helpers and utilities.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    ~supervisely.pointcloud.pointcloud

Pointcloud Annotation
---------------------
Data model for pointcloud annotations (figures, objects, tags, episodes).

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ~supervisely.pointcloud_annotation.pointcloud_annotation.PointcloudAnnotation
    ~supervisely.pointcloud_annotation.pointcloud_figure.PointcloudFigure
    ~supervisely.pointcloud_annotation.pointcloud_object.PointcloudObject
    ~supervisely.pointcloud_annotation.pointcloud_object_collection.PointcloudObjectCollection
    ~supervisely.pointcloud_annotation.pointcloud_tag.PointcloudTag
    ~supervisely.pointcloud_annotation.pointcloud_tag_collection.PointcloudTagCollection
    ~supervisely.pointcloud_annotation.pointcloud_episode_annotation.PointcloudEpisodeAnnotation
    ~supervisely.pointcloud_annotation.pointcloud_episode_frame.PointcloudEpisodeFrame
    ~supervisely.pointcloud_annotation.pointcloud_episode_frame_collection.PointcloudEpisodeFrameCollection
    ~supervisely.pointcloud_annotation.pointcloud_episode_tag.PointcloudEpisodeTag
    ~supervisely.pointcloud_annotation.pointcloud_episode_tag_collection.PointcloudEpisodeTagCollection


Pointcloud Episodes
-------------------
Helpers and data structures for pointcloud episodes.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    ~supervisely.pointcloud_episodes.pointcloud_episodes

Project
-------
Working with Supervisely projects on disk.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ~supervisely.project.project.Project
    ~supervisely.project.project_type.ProjectType
    ~supervisely.project.project_meta.ProjectMeta
    ~supervisely.project.project_meta.ProjectMetaJsonFields
    ~supervisely.project.project.Dataset
    ~supervisely.project.project.DatasetDict
    ~supervisely.project.project.OpenMode
    ~supervisely.project.project.ItemPaths
    ~supervisely.project.project.ItemInfo
    ~supervisely.project.video_project.VideoProject
    ~supervisely.project.video_project.VideoDataset
    ~supervisely.project.video_project.VideoItemPaths
    ~supervisely.project.pointcloud_project.PointcloudProject
    ~supervisely.project.pointcloud_project.PointcloudDataset
    ~supervisely.project.volume_project.VolumeProject
    ~supervisely.project.volume_project.VolumeDataset
    ~supervisely.project.data_version.VersionInfo
    ~supervisely.project.data_version.DataVersion

Task
----
Constants defining the directories where a plugin should expect the input and output data to be, plus helpers to stream progress data from a running plugin back to the web instance.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ~supervisely.task.progress.Progress

User
----
User-related data structures and constants.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ~supervisely.user.user.UserRoleName

Video
-----
Core video-related helpers and utilities.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    ~supervisely.video.video

Video Annotation
----------------
Data model for video annotations (frames, objects, figures, tags).

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ~supervisely.video_annotation.video_annotation.VideoAnnotation
    ~supervisely.video_annotation.frame.Frame
    ~supervisely.video_annotation.frame_collection.FrameCollection
    ~supervisely.video_annotation.key_id_map.KeyIdMap
    ~supervisely.video_annotation.video_figure.VideoFigure
    ~supervisely.video_annotation.video_object.VideoObject
    ~supervisely.video_annotation.video_object_collection.VideoObjectCollection
    ~supervisely.video_annotation.video_tag.VideoTag
    ~supervisely.video_annotation.video_tag_collection.VideoTagCollection

Volume
------
Core volume-related helpers and utilities.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-module-template.rst

    ~supervisely.volume.volume

Volume Annotation
-----------------
Data model for volume annotations (objects, figures, tags, and slices).

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ~supervisely.volume_annotation.volume_annotation.VolumeAnnotation
    ~supervisely.volume_annotation.volume_figure.VolumeFigure
    ~supervisely.volume_annotation.volume_object.VolumeObject
    ~supervisely.volume_annotation.volume_object_collection.VolumeObjectCollection
    ~supervisely.volume_annotation.volume_tag.VolumeTag
    ~supervisely.volume_annotation.volume_tag_collection.VolumeTagCollection
    ~supervisely.volume_annotation.plane.Plane
    ~supervisely.volume_annotation.slice.Slice

Utility Functions
-----------------
A collection of useful utility functions for common tasks in the Supervisely SDK.

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-function-template.rst

    ~supervisely.project.download.download_fast

Training
--------
High-level wrappers and helpers for building training applications (GUI, data prep, artifacts upload, benchmarking).

.. autosummary::
    :toctree: sdk
    :nosignatures:
    :template: autosummary/custom-class-template.rst

    ~supervisely.nn.training.train_app.TrainApp
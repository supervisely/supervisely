from supervisely.app.widgets.widget import ConditionalWidget, ConditionalItem, DynamicWidget
from supervisely.app.widgets.widget import Widget, generate_id
from supervisely.app.widgets.radio_table.radio_table import RadioTable
from supervisely.app.widgets.notification_box.notification_box import NotificationBox
from supervisely.app.widgets.done_label.done_label import DoneLabel
from supervisely.app.widgets.sly_tqdm.sly_tqdm import SlyTqdm, Progress
from supervisely.app.widgets.grid_gallery.grid_gallery import GridGallery
from supervisely.app.widgets.classes_table.classes_table import ClassesTable
from supervisely.app.widgets.classic_table.classic_table import ClassicTable
from supervisely.app.widgets.confusion_matrix.confusion_matrix import ConfusionMatrix
from supervisely.app.widgets.project_selector.project_selector import ProjectSelector
from supervisely.app.widgets.element_button.element_button import ElementButton
from supervisely.app.widgets.button.button import Button
from supervisely.app.widgets.project_thumbnail.project_thumbnail import ProjectThumbnail
from supervisely.app.widgets.apexchart.apexchart import Apexchart
from supervisely.app.widgets.line_chart.line_chart import LineChart
from supervisely.app.widgets.grid_chart.grid_chart import GridChart
from supervisely.app.widgets.scatter_chart.scatter_chart import ScatterChart
from supervisely.app.widgets.heatmap_chart.heatmap_chart import HeatmapChart
from supervisely.app.widgets.treemap_chart.treemap_chart import TreemapChart
from supervisely.app.widgets.table.table import Table
from supervisely.app.widgets.labeled_image.labeled_image import LabeledImage
from supervisely.app.widgets.text.text import Text
from supervisely.app.widgets.sidebar.sidebar import Sidebar
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.card.card import Card
from supervisely.app.widgets.select.select import Select, SelectString
from supervisely.app.widgets.menu.menu import Menu
from supervisely.app.widgets.field.field import Field
from supervisely.app.widgets.input_number.input_number import InputNumber
from supervisely.app.widgets.video.video import Video
from supervisely.app.widgets.object_class_view.object_class_view import ObjectClassView
from supervisely.app.widgets.checkbox.checkbox import Checkbox
from supervisely.app.widgets.grid.grid import Grid
from supervisely.app.widgets.object_classes_list.object_classes_list import (
    ObjectClassesList,
)
from supervisely.app.widgets.empty.empty import Empty
from supervisely.app.widgets.one_of.one_of import OneOf
from supervisely.app.widgets.flexbox.flexbox import Flexbox
from supervisely.app.widgets.input.input import Input
from supervisely.app.widgets.select_team.select_team import SelectTeam
from supervisely.app.widgets.select_workspace.select_workspace import SelectWorkspace
from supervisely.app.widgets.select_project.select_project import SelectProject
from supervisely.app.widgets.select_dataset.select_dataset import SelectDataset
from supervisely.app.widgets.select_item.select_item import SelectItem
from supervisely.app.widgets.select_app_session.select_app_session import SelectAppSession
from supervisely.app.widgets.select_cuda.select_cuda import SelectCudaDevice
from supervisely.app.widgets.identity.identity import Identity
from supervisely.app.widgets.dataset_thumbnail.dataset_thumbnail import DatasetThumbnail
from supervisely.app.widgets.select_tag_meta.select_tag_meta import SelectTagMeta
from supervisely.app.widgets.video_thumbnail.video_thumbnail import VideoThumbnail
from supervisely.app.widgets.tabs.tabs import Tabs
from supervisely.app.widgets.radio_tabs.radio_tabs import RadioTabs
from supervisely.app.widgets.train_val_splits.train_val_splits import TrainValSplits
from supervisely.app.widgets.editor.editor import Editor
from supervisely.app.widgets.textarea.textarea import TextArea
from supervisely.app.widgets.destination_project.destination_project import DestinationProject
from supervisely.app.widgets.image.image import Image
from supervisely.app.widgets.random_splits_table.random_splits_table import RandomSplitsTable
from supervisely.app.widgets.video_player.video_player import VideoPlayer
from supervisely.app.widgets.radio_group.radio_group import RadioGroup
from supervisely.app.widgets.switch.switch import Switch
from supervisely.app.widgets.input_tag.input_tag import InputTag

from supervisely.app.widgets.file_viewer.file_viewer import FileViewer
from supervisely.app.widgets.switch.switch import Switch
from supervisely.app.widgets.folder_thumbnail.folder_thumbnail import FolderThumbnail
from supervisely.app.widgets.file_thumbnail.file_thumbnail import FileThumbnail
from supervisely.app.widgets.model_info.model_info import ModelInfo
from supervisely.app.widgets.match_tags_or_classes.match_tags_or_classes import (
    MatchTagMetas,
    MatchObjClasses,
)
from supervisely.app.widgets.match_datasets.match_datasets import MatchDatasets
from supervisely.app.widgets.line_plot.line_plot import LinePlot
from supervisely.app.widgets.grid_plot.grid_plot import GridPlot
from supervisely.app.widgets.binded_input_number.binded_input_number import BindedInputNumber
from supervisely.app.widgets.augmentations.augmentations import Augmentations, AugmentationsWithTabs

from supervisely.app.widgets.tabs_dynamic.tabs_dynamic import TabsDynamic
from supervisely.app.widgets.stepper.stepper import Stepper
from supervisely.app.widgets.slider.slider import Slider
from supervisely.app.widgets.copy_to_clipboard.copy_to_clipboard import CopyToClipboard
from supervisely.app.widgets.file_storage_upload.file_storage_upload import FileStorageUpload
from supervisely.app.widgets.image_region_selector.image_region_selector import ImageRegionSelector
from supervisely.app.widgets.collapse.collapse import Collapse
from supervisely.app.widgets.team_files_selector.team_files_selector import TeamFilesSelector
from supervisely.app.widgets.icons.icons import Icons
from supervisely.app.widgets.badge.badge import Badge
from supervisely.app.widgets.date_picker.date_picker import DatePicker
from supervisely.app.widgets.datetime_picker.datetime_picker import DateTimePicker
from supervisely.app.widgets.transfer.transfer import Transfer
from supervisely.app.widgets.task_logs.task_logs import TaskLogs
from supervisely.app.widgets.reloadable_area.reloadable_area import ReloadableArea
from supervisely.app.widgets.image_pair_sequence.image_pair_sequence import ImagePairSequence
from supervisely.app.widgets.markdown.markdown import Markdown
from supervisely.app.widgets.class_balance.class_balance import ClassBalance
from supervisely.app.widgets.image_slider.image_slider import ImageSlider
from supervisely.app.widgets.rate.rate import Rate
from supervisely.app.widgets.carousel.carousel import Carousel
from supervisely.app.widgets.dropdown.dropdown import Dropdown
from supervisely.app.widgets.pie_chart.pie_chart import PieChart
from supervisely.app.widgets.timeline.timeline import Timeline
from supervisely.app.widgets.nodes_flow.nodes_flow import NodesFlow
from supervisely.app.widgets.dialog.dialog import Dialog
from supervisely.app.widgets.draggable.draggable import Draggable
from supervisely.app.widgets.tooltip.tooltip import Tooltip
from supervisely.app.widgets.image_annotation_preview.image_annotation_preview import (
    ImageAnnotationPreview,
)
from supervisely.app.widgets.tag_meta_view.tag_meta_view import TagMetaView
from supervisely.app.widgets.tag_metas_list.tag_metas_list import TagMetasList

from supervisely.app.widgets.color_picker.color_picker import ColorPicker
from supervisely.app.widgets.pagination.pagination import Pagination
from supervisely.app.widgets.cascader.cascader import Cascader
from supervisely.app.widgets.time_picker.time_picker import TimePicker
from supervisely.app.widgets.fast_table.fast_table import FastTable
from supervisely.app.widgets.element_tag.element_tag import ElementTag
from supervisely.app.widgets.element_tags_list.element_tags_list import ElementTagsList
from supervisely.app.widgets.compare_annotations.compare_annotations import CompareAnnotations

from supervisely.app.widgets.circle_progress.circle_progress import CircleProgress
from supervisely.app.widgets.classes_color_mapping.classes_color_mapping import ClassesColorMapping
from supervisely.app.widgets.classes_mapping.classes_mapping import ClassesMapping
from supervisely.app.widgets.classes_mapping_preview.classes_mapping_preview import (
    ClassesMappingPreview,
)
from supervisely.app.widgets.classes_list_selector.classes_list_selector import ClassesListSelector
from supervisely.app.widgets.classes_list_preview.classes_list_preview import ClassesListPreview
from supervisely.app.widgets.tags_list_selector.tags_list_selector import TagsListSelector
from supervisely.app.widgets.tags_list_preview.tags_list_preview import TagsListPreview
from supervisely.app.widgets.members_list_selector.members_list_selector import MembersListSelector
from supervisely.app.widgets.members_list_preview.members_list_preview import MembersListPreview
from supervisely.app.widgets.custom_models_selector.custom_models_selector import (
    CustomModelsSelector,
)
from supervisely.app.widgets.agent_selector.agent_selector import AgentSelector
from supervisely.app.widgets.iframe.iframe import IFrame
from supervisely.app.widgets.pretrained_models_selector.pretrained_models_selector import (
    PretrainedModelsSelector,
)

from supervisely.app.widgets.tags_table.tags_table import TagsTable
from supervisely.app.widgets.checkbox_field.checkbox_field import CheckboxField
from supervisely.app.widgets.tree_select.tree_select import TreeSelect
from supervisely.app.widgets.select_dataset_tree.select_dataset_tree import SelectDatasetTree
from supervisely.app.widgets.grid_gallery_v2.grid_gallery_v2 import GridGalleryV2
from supervisely.app.widgets.report_thumbnail.report_thumbnail import ReportThumbnail
from supervisely.app.widgets.experiment_selector.experiment_selector import ExperimentSelector
from supervisely.app.widgets.bokeh.bokeh import Bokeh
from supervisely.app.widgets.run_app_button.run_app_button import RunAppButton
from supervisely.app.widgets.select_collection.select_collection import SelectCollection
from supervisely.app.widgets.sampling.sampling import Sampling
from supervisely.app.widgets.input_tag_list.input_tag_list import InputTagList
from supervisely.app.widgets.deploy_model.deploy_model import DeployModel
from supervisely.app.widgets.dropdown_checkbox_selector.dropdown_checkbox_selector import (
    DropdownCheckboxSelector,
)
from supervisely.app.widgets.ecosystem_model_selector.ecosystem_model_selector import (
    EcosystemModelSelector,
)
from supervisely.app.widgets.heatmap.heatmap import Heatmap

from typing import Dict, Optional, Union

from supervisely.app.content import StateJson
from supervisely.app.widgets import Widget


class Embeddings(Widget):
    def __init__(
        self,
        pointcloud_url: Optional[str] = None,
        atlas_manifest: Optional[Dict[str, Union[int, str]]] = None,
        enable_zoom: Optional[bool] = True,
        width: Optional[Union[int, str]] = 800,
        height: Optional[Union[int, str]] = 600,
        widget_id: Optional[str] = None,
    ):
        self._pointcloud_url = pointcloud_url
        self._atlas_manifest = atlas_manifest
        self._style = self._get_style(width, height)

        self._options = {
            "enableZoom": enable_zoom,
        }

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _get_size(self, size: Union[int, str]) -> str:
        if isinstance(size, int) or size.isdigit():
            return f"{size}px"
        elif size.endswith("%") or size.endswith("px"):
            return size
        else:
            raise ValueError(
                f"Size must be either in pixels or percentage, instead got: {size}. "
                "Correct examples: 800, '800px', '80%'."
            )

    def _get_style(self, width: Union[str, int], height: Union[str, int]) -> str:
        return f"width: {self._get_size(width)}; height: {self._get_size(height)}"

    @property
    def pointcloud_url(self):
        return self._pointcloud_url

    @pointcloud_url.setter
    def pointcloud_url(self, url: str) -> None:
        self._pointcloud_url = url
        self.update_state()
        StateJson().send_changes()

    @property
    def atlas_manifest(self):
        return self._atlas_manifest

    @atlas_manifest.setter
    def atlas_manifest(self, manifest: Dict) -> None:
        self._atlas_manifest = manifest
        self.update_state()
        StateJson().send_changes()

    def set_pointcloud_url(self, url: str) -> None:
        self.pointcloud_url = url

    def set_atlas_manifest(self, atlas_manifest: Dict) -> None:
        self.atlas_manifest = atlas_manifest

    def set_data(self, pointcloud_url: str, atlas_manifest: Dict) -> None:
        self.set_pointcloud_url(pointcloud_url)
        self.set_atlas_manifest(atlas_manifest)

    def get_json_state(self):
        return {
            "pointCloudUrl": self.pointcloud_url,
            "atlasManifest": self.atlas_manifest,
            "options": self._options,
        }

    def get_json_data(self):
        return {
            "style": self._style,
        }

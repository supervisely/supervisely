from typing import Dict, List, Optional, Tuple, Union

from supervisely._utils import generate_free_name

from supervisely.convert.base_converter import AvailablePointcloudEpisodesConverters
from supervisely.convert.pointcloud.seyond_pcap.seyond_pcap_converter import (
    SeyondPcapConverter,
)
from supervisely.convert.pointcloud.seyond_pcap.seyond_pcap_helper import to_iso_name
from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import (
    PointcloudEpisodeConverter,
)
from supervisely.project.project_settings import LabelingInterface


class SeyondPcapEpisodesConverter(SeyondPcapConverter, PointcloudEpisodeConverter):
    """Imports Seyond lidar pcap captures as point cloud episodes.

    The captures of each sensor are chained by time: the next capture continues the episode
    when it starts within ``episodes.maxGapSec`` (1 s by default) after the previous one's
    last frame starts, and while the episode stays within ``episodes.maxFrames``
    (``sensors.json``). Captures that overlap in time start separate episodes. A capture is
    never split. Each episode is a dataset; one episode goes to the target dataset.
    """

    def __init__(
        self,
        input_data: str,
        labeling_interface: Optional[Union[LabelingInterface, str]],
        upload_as_links: bool,
        remote_files_map: Optional[Dict[str, str]] = None,
        team_files_id_map: Optional[Dict[str, str]] = None,
    ):
        """See :class:`~supervisely.convert.base_converter.BaseConverter` for params."""
        super().__init__(
            input_data, labeling_interface, upload_as_links, remote_files_map, team_files_id_map
        )
        self._is_pcd_episode = True

    def __str__(self) -> str:
        return AvailablePointcloudEpisodesConverters.SEYOND_PCAP

    def _group_items(self) -> List[Tuple[str, List[SeyondPcapConverter.Item]]]:
        episodes: List[List[SeyondPcapConverter.Item]] = []
        for item in self._items:  # sorted by sensor, then start time
            current = episodes[-1] if episodes else None
            if (
                current is not None
                and current[-1].sensor == item.sensor
                # a capture that overlaps the previous one (the same file twice, or a
                # re-recording) never continues it: the tool numbers frames per file
                and 0 < item.start_time - current[-1].end_time <= self._config.max_gap_sec
                and (
                    self._config.max_frames is None
                    or sum(i.frame_count for i in current) + item.frame_count
                    <= self._config.max_frames
                )
            ):
                current.append(item)
            else:
                episodes.append([item])

        per_sensor: Dict[str, int] = {}
        for episode in episodes:
            per_sensor[episode[0].sensor] = per_sensor.get(episode[0].sensor, 0) + 1
        groups, used_names = [], set()
        for episode in episodes:
            sensor = episode[0].sensor
            name = sensor
            if per_sensor[sensor] > 1:
                name = f"{sensor} {to_iso_name(episode[0].start_time)}"
            name = generate_free_name(used_names, name, extend_used_names=True)
            groups.append((name, episode))
        return groups

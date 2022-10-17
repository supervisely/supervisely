from supervisely._utils import is_development, abs_url


def get_labeling_tool_url(dataset_id, pointcloud_id):
    res = f"/app/point-clouds-tracking/?datasetId={dataset_id}&pointCloudId={pointcloud_id}"
    if is_development():
        res = abs_url(res)
    return res


def get_labeling_tool_link(url, name="open in labeling tool"):
    return f'<a href="{url}" rel="noopener noreferrer" target="_blank">{name}<i class="zmdi zmdi-open-in-new" style="margin-left: 5px"></i></a>'

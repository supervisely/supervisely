"""Processing of pointcloud episodes"""

from supervisely._utils import is_development, abs_url


def get_labeling_tool_url(dataset_id, pointcloud_id):
    """
    Get the URL for the labeling tool with the specified dataset ID and poin tcloud ID.

    :param dataset_id: Dataset ID in Supervisely.
    :type dataset_id: int
    :param pointcloud_id: Point cloud ID in Supervisely.
    :type pointcloud_id: int
    :return: URL for the labeling tool with the specified dataset ID and point cloud ID
    :rtype: str
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # You can connect to API directly
        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Or you can use API from environment
        os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
        os.environ['API_TOKEN'] = 'Your Supervisely API Token'
        api = sly.Api.from_env()

        pointcloud_id = 19373403
        pcd_info = api.pointcloud_episodes.get_info_by_id(pointcloud_id)
        url = sly.pointcloud_episodes.get_labeling_tool_url(pcd_info.dataset_id, pcd_info.id)

        print(url)
        # Output: 
        # https://dev.supervise.ly/app/point-clouds-tracking/?datasetId=55875&pointCloudId=19373403
    """

    res = f"/app/point-clouds-tracking/?datasetId={dataset_id}&pointCloudId={pointcloud_id}"
    if is_development():
        res = abs_url(res)
    return res


def get_labeling_tool_link(url, name="open in labeling tool"):
    """
    Get HTML link to the labeling tool with the specified URL and name.

    :param url: URL of the labeling tool.
    :type url: str
    :param name: Name of the link, default is "open in labeling tool".
    :type name: str
    :return: HTML link to the labeling tool with the specified URL and name.
    :rtype: str
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # You can connect to API directly
        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Or you can use API from environment
        os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
        os.environ['API_TOKEN'] = 'Your Supervisely API Token'
        api = sly.Api.from_env()

        pointcloud_id = 19373403
        pcd_info = api.pointcloud.get_info_by_id(pointcloud_id)
        url = sly.pointcloud.get_labeling_tool_url(pcd_info.dataset_id, pcd_info.id)
        name = "my link"

        link = sly.pointcloud.get_labeling_tool_link(url, name)

        print(link)
        # Output: 
        # <a 
        #     href="https://dev.supervise.ly/app/point-clouds/?datasetId=55875&pointCloudId=19373403"
        #     rel="noopener noreferrer"
        #     target="_blank"
        # >
        #     my link<i class="zmdi zmdi-open-in-new" style="margin-left: 5px"></i>
        # </a>
    """

    return f'<a href="{url}" rel="noopener noreferrer" target="_blank">{name}<i class="zmdi zmdi-open-in-new" style="margin-left: 5px"></i></a>'

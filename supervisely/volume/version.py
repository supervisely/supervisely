from supervisely.project.data_version import DataVersion


class VolumeDataVersion(DataVersion):
    """
    DataVersion tailored for Volume projects: stores only annotation blobs
    serialized with pyarrow to keep versions lightweight.
    """
    @property
    def project_cls(self):
        from supervisely.project.volume_project import VolumeProject
        return VolumeProject
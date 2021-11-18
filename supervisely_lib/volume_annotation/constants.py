ID = 'id'
KEY = 'key'
TAGS = 'tags'
INDEX = 'index'
OBJECTS = 'objects'
FIGURES = 'figures'
OBJECT_KEY = 'objectKey'

VOLUME_ID = 'volumeId'
VOLUME_NAME = 'volumeName'
VOLUME_META = 'volumeMeta'

PLANES = 'planes'
SLICES = 'slices'
SLICE_INDEX = 'sliceIndex'

NAME = 'name'
AXIAL = 'axial'
NORMAL = 'normal'
CORONAL = 'coronal'
SAGITTAL = 'sagittal'

META = 'meta'
DESCRIPTION = 'description'

PLANE_NAMES = [AXIAL, SAGITTAL, CORONAL]
PLANE_NORMALS = {
    SAGITTAL: {"x": 1, "y": 0, "z": 0},
    CORONAL: {"x": 0, "y": 1, "z": 0},
    AXIAL: {"x": 0, "y": 0, "z": 1},
}

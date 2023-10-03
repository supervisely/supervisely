import os
import numpy as np
from dotenv import load_dotenv
import supervisely as sly

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

#################################### TEST for #################################
############################### download_volume_project #######################
sly.fs.remove_dir("STL_UPLOAD_TEST")
sly.download_volume_project(api, 28637, "STL_UPLOAD_TEST", log_progress=True)

#################################### TEST for #################################
################################ upload_volume_project  #######################
################################### _append_bulk_mask3d #######################
######################################## upload_sf_geometries #################
sly.upload_volume_project("STL_UPLOAD_TEST", api, 1012, "STL_UPLOAD_TEST", True)

#################################### TEST for #################################
############################### project.append_classes  #######################
volume_info = api.volume.get_info_by_id(24361195)
lung1_obj_class = sly.ObjClass("lung_1", sly.Mask3D)
lung2_obj_class = sly.ObjClass("lung_2", sly.Mask3D)
lung3_obj_class = sly.ObjClass("lung_3", sly.Mask3D)
bung_obj_class = sly.ObjClass("rectangle", sly.Rectangle)
poly_obj_class = sly.ObjClass("poly", sly.Polygon)
classes = [lung1_obj_class, bung_obj_class, lung2_obj_class, poly_obj_class, lung3_obj_class]
api.project.append_classes(id=28639, classes=classes) 

############################# Mask3D.from_bytes #########################

file_path = "data/mask/lung.nrrd" # from tutorial
with open(file_path, "rb") as file:
    geometry_bytes = file.read()
mask_3d_frombytes = sly.Mask3D.from_bytes(geometry_bytes)

############################# VolumeObject.__init__ ###########################
############################# Mask3D.create_from_file #########################
lung1 = sly.VolumeObject(lung1_obj_class, mask_3d=file_path) # from file
lung2 = sly.VolumeObject(
    lung2_obj_class, mask_3d=np.random.randint(2, size=(50, 50, 50), dtype=np.bool_) # from NumPy
)

############################# VolumeFigure.__init__ ###########################
lung3 = sly.VolumeObject(lung2_obj_class, mask_3d=mask_3d_frombytes) # from Mask3D
rect = sly.VolumeObject(bung_obj_class)
poly = sly.VolumeObject(poly_obj_class)
objects = sly.VolumeObjectCollection([lung1, rect, lung2, poly, lung3])

############################# annotation.append_objects #######################
api.volume.annotation.append_objects(volume_info.id, objects)


print("done")

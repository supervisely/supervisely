import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from supervisely.project.project import Project
from supervisely.sly_logger import logger


# function for data preprocessing
def prepare_segmentation_data(source_project_dir, output_project_dir, palette):
    if os.path.exists(output_project_dir):
        logger.info(f"Preprocessed data already exists in {output_project_dir} directory")
        return
    else:
        os.makedirs(output_project_dir)

        ann_dir = "seg"

        temp_project_seg_dir = source_project_dir + "_temp"
        if not os.path.exists(temp_project_seg_dir):
            Project.to_segmentation_task(
                source_project_dir,
                temp_project_seg_dir,
            )

        datasets = os.listdir(temp_project_seg_dir)
        for dataset in datasets:
            if not os.path.isdir(os.path.join(temp_project_seg_dir, dataset)):
                continue
            # convert masks to required format and save to general ann_dir
            mask_files = os.listdir(os.path.join(temp_project_seg_dir, dataset, ann_dir))
            for mask_file in tqdm(mask_files, desc="Preparing segmentation data..."):
                mask = cv2.imread(os.path.join(temp_project_seg_dir, dataset, ann_dir, mask_file))[
                    :, :, ::-1
                ]
                result = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
                # human masks to machine masks
                for color_idx, color in enumerate(palette):
                    colormap = np.where(np.all(mask == color, axis=-1))
                    result[colormap] = color_idx
                if mask_file.count(".png") > 1:
                    mask_file = mask_file[:-4]
                cv2.imwrite(os.path.join(output_project_dir, mask_file), result)

        shutil.rmtree(temp_project_seg_dir)

# coding: utf-8

import os.path as osp

import cv2
from supervisely_lib import logger
import supervisely_lib as sly


# In this example we will read Supervisely project from disk and draw annotations on corresponding images.


def main():
    logger.info('Hello world.')

    # It isn't necessary, but let's suppose that our data will be stored as for Supervisely task:
    # input in '/sly_task_data/data` and results in '/sly_task_data/results'.
    # So TaskPaths provides the paths.
    task_paths = sly.TaskPaths()

    project_dir = task_paths.project_dir  # the paths includes project name

    project_meta = sly.ProjectMeta.from_dir(project_dir)
    # Now we've read meta of input project.
    logger.info('Input project meta: {} class(es).'.format(len(project_meta.classes)))

    project_fs = sly.ProjectFS.from_disk(*sly.ProjectFS.split_dir_project(project_dir))
    # Now we've read project structure.
    logger.info('Input project: "{}" contains {} dataset(s) and {} image(s).'.format(
        project_fs.pr_structure.name,
        len(project_fs.pr_structure.datasets),
        project_fs.image_cnt
    ))

    # prepare color mapping
    color_mapping = {}
    for cls_descr in project_meta.classes:
        color_s = cls_descr.get('color')
        if color_s is not None:
            color = sly.hex2rgb(color_s)  # use color from project meta if exists
        else:
            color = sly.get_random_color()  # or use random color otherwise
        color_mapping[cls_descr['title']] = color

    # enumerate all input samples (image/annotation pairs)
    for item_descr in project_fs:
        logger.info('Processing input sample',
                    extra={'dataset': item_descr.ds_name, 'image_name': item_descr.image_name})

        # Open image
        img = cv2.imread(item_descr.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # to work with r-g-b colors

        # And read corresponding annotation.
        ann_packed = sly.json_load(item_descr.ann_path)
        ann = sly.Annotation.from_packed(ann_packed, project_meta)

        # Draw annotations on image
        for fig in ann['objects']:
            color = color_mapping.get(fig.class_title)
            fig.draw(img, color)
            # Note that this method draws lines with width 1, and points as single pixels.

        # Save image. Please note that we just save images, not new Supervisely project.
        src_image_ext = item_descr.ia_data['image_ext']  # let's preserve source image format (by extension)
        out_fpath = osp.join(
            task_paths.results_dir, item_descr.project_name, item_descr.ds_name, item_descr.image_name + src_image_ext
        )
        sly.ensure_base_path(out_fpath)  # create intermediate dirs if required

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # to work with r-g-b colors
        cv2.imwrite(out_fpath, img)  # write image

    logger.info('Done.')


if __name__ == '__main__':
    main()

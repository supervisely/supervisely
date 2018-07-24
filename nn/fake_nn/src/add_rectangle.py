# coding: utf-8

from copy import deepcopy

from supervisely_lib import logger
import supervisely_lib as sly


# In this example we will read Supervisely project from disk, add rectangle of new class to each image annotation
# and store result as a new Supervisely project.


def main():
    logger.info('Hello world.')

    # It isn't necessary, but let's suppose that our data will be stored as for Supervisely task:
    # input in '/sly_task_data/data` and results in '/sly_task_data/results'.
    # So TaskPaths provides the paths.
    task_paths = sly.TaskPaths()

    in_pr_dir = task_paths.project_dir  # the paths includes project name

    in_pr_meta = sly.ProjectMeta.from_dir(in_pr_dir)
    # Now we've read meta of input project.
    logger.info('Input project meta: {} class(es).'.format(len(in_pr_meta.classes)))

    in_pr_fs = sly.ProjectFS.from_disk(*sly.ProjectFS.split_dir_project(in_pr_dir))
    # Now we've read project structure.
    logger.info('Input project: "{}" contains {} dataset(s) and {} image(s).'.format(
        in_pr_fs.pr_structure.name,
        len(in_pr_fs.pr_structure.datasets),
        in_pr_fs.image_cnt
    ))

    # It's convenient to create output project structure and store source file paths in ia_data.
    out_pr_structure = sly.ProjectStructure('my_new_project')  # rename project... just for fun
    for item_descr in in_pr_fs:  # iterate over input project
        new_ia_data = {
            'src_ann_path': item_descr.ann_path,
            'src_img_path': item_descr.img_path,
            **item_descr.ia_data  # contains 'image_ext' which is required to write images
        }
        out_pr_structure.add_item(item_descr.ds_name, item_descr.image_name, new_ia_data)
    # ProjectFS will provide out file paths
    out_pr_fs = sly.ProjectFS(task_paths.results_dir, out_pr_structure)

    # We will add the rectangle to each annotation.
    new_class_title = 'new-region'
    rect_to_add = sly.Rect(left=20, top=20, right=50, bottom=100)

    # Ok, start processing.
    out_pr_fs.make_dirs()  # create all directories required for writing
    for item_descr in out_pr_fs:  # iterate over output project
        logger.info('Processing sample',
                    extra={'dataset': item_descr.ds_name, 'image_name': item_descr.image_name})

        # Copy image unchanged.
        sly.copy_file(item_descr.ia_data['src_img_path'], item_descr.img_path)

        # Read annotation.
        ann_packed = sly.json_load(item_descr.ia_data['src_ann_path'])
        ann = sly.Annotation.from_packed(ann_packed, in_pr_meta)

        # Add new figure to the annotation.
        # Method to construct figures returns iterable of new figures.
        # (e.g., line cropped with image bounds may produce some lines), but here we'll get not more than one figure
        # ...or no figures if image is less than 20x20.
        new_figures = sly.FigureRectangle.from_rect(new_class_title, ann.image_size_wh, rect_to_add)
        ann['objects'].extend(new_figures)

        # Save annotation.
        sly.json_dump(ann.pack(), item_descr.ann_path)

    # OK, and don't forget to create and save output project meta.
    # We'll save given data and add new class with shape "rectangle".
    out_pr_meta = deepcopy(in_pr_meta)
    out_pr_meta.classes.add({'title': new_class_title, 'shape': 'rectangle', 'color': '#FFFF00'})
    # Then store the meta.
    out_pr_meta.to_dir(out_pr_fs.project_path)

    logger.info('Done.')


if __name__ == '__main__':
    main()

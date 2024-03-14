import os

from dotenv import load_dotenv

import supervisely as sly
from supervisely.api.entity_annotation.figure_api import FigureApi
from supervisely.api.image_api import ImageApi
from supervisely.api.module_api import ApiField

if sly.is_development():
    load_dotenv(os.path.expanduser("~/ninja.env"))
    # load_dotenv("local.env")
api = sly.Api()


TEAM_ID = 9
PROJECT_IDS = [223, 2070, 1574]

import time

# Your code block here


def images_list(project_id):

    project = api.project.get_info_by_id(project_id)
    print("images.list: ", project.name, "\n-------------")

    for dataset in api.dataset.get_list(project_id):
        start_time = time.time()
        # api.image.get_list(dataset.id)
        ImageApi(api).get_list_all_pages(
            "images.list",
            {
                ApiField.DATASET_ID: dataset.id,
                ApiField.PER_PAGE: 10000,
            },
        )
        end_time = time.time()

        elapsed_time = end_time - start_time

        print(
            project.name,
            "| Dataset:",
            dataset.name,
            "| Num Images:",
            dataset.images_count,
            "| Dataset weight GB:",
            round((float(dataset.size) / 1024 / 1024 / 1024), 2),
        )
        print("Elapsed Time:", elapsed_time, "seconds\n")


def figure_list(project_id):
    fields = [
        "id",
        "createdAt",
        "updatedAt",
        "imageId",
        "objectId",
        "classId",
        "projectId",
        # "datasetId",
        # "geometryType",
        # "geometry",
        "tags",
        "meta",
    ]
    project = api.project.get_info_by_id(project_id)
    print("figures.list: ", project.name, "\n-------------")

    for dataset in api.dataset.get_list(project_id):
        start_time = time.time()
        figures_infos = FigureApi(api).get_list_all_pages(
            "figures.list",
            {
                ApiField.DATASET_ID: dataset.id,
                ApiField.FIELDS: fields,
            },
        )
        end_time = time.time()

        elapsed_time = end_time - start_time

        print(
            project.name,
            "| Dataset:",
            dataset.name,
            "| Num Images:",
            dataset.images_count,
            "| Dataset weight GB:",
            round((float(dataset.size) / 1024 / 1024 / 1024), 2),
        )
        print("Elapsed Time:", elapsed_time, "seconds\n")


if __name__ == "__main__":
    for id in PROJECT_IDS:
        images_list(id)
        figure_list(id)

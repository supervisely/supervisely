import csv
import os
import os.path
from typing import Tuple

import pandas as pd

from supervisely import ProjectMeta, TagMeta, TagMetaCollection, TagValueType, logger
from supervisely.io.fs import silent_remove

DEFAULT_DELIMITER = ";"
INCORRECT_COLUMN_DELIMITER = ","
TAGS_DELIMITER = ","
possible_image_url_col_names = ["url"]
possible_image_path_col_names = ["path"]
possible_tag_col_names = ["tag"]

# ------------------------------------------- Convert ------------------------------------------- #


def rename_in_json(ann_json, renamed_classes=None, renamed_tags=None):
    if renamed_classes:
        for obj in ann_json["objects"]:
            obj["classTitle"] = renamed_classes.get(obj["classTitle"], obj["classTitle"])
    if renamed_tags:
        for tag in ann_json["tags"]:
            tag["name"] = renamed_tags.get(tag["name"], tag["name"])
    return ann_json


def convert_txt_to_csv(txt_file_path: str, csv_file_path: str, delimiter: str = r"\s{2,}|\t+|;"):
    data = pd.read_csv(txt_file_path, delimiter=delimiter, engine="python")
    data.to_csv(csv_file_path, index=False, sep=DEFAULT_DELIMITER)


def convert_tsv_to_csv(tsv_file_path: str, csv_file_path: str, delimiter: str = DEFAULT_DELIMITER):
    data = pd.read_csv(tsv_file_path, delimiter="\t")
    data.to_csv(csv_file_path, index=False, sep=delimiter)


# ------------------------------------------- Validate ------------------------------------------- #


def flat_tag_list(total_tags):
    tags = []
    for tag in total_tags:
        if tag is None or tag == "":
            continue
        else:
            if TAGS_DELIMITER in tag:
                tag = tag.split(TAGS_DELIMITER)
            tags.append(tag)

    total_tags = []
    for sublist in tags:
        if type(sublist) == str:
            total_tags.append(sublist)
        else:
            for tag in sublist:
                if tag != "":
                    total_tags.append(tag)

    total_tags = list(set(total_tags))
    return total_tags


def create_project_meta_from_csv_tags(total_tags):
    tag_metas = []
    for tag_name in total_tags:
        tag_meta = TagMeta(tag_name, TagValueType.NONE)
        tag_metas.append(tag_meta)
    tag_meta_col = TagMetaCollection(tag_metas)
    project_meta = ProjectMeta(tag_metas=tag_meta_col)
    return project_meta


def check_column_names(col_names_validate):
    if not any(
        image_url_name in possible_image_url_col_names for image_url_name in col_names_validate
    ) and not any(
        image_path_name in possible_image_path_col_names for image_path_name in col_names_validate
    ):
        raise ValueError(
            f"Invalid column name for image: '{col_names_validate}' in the first line. "
            f"Please use: '{ possible_image_url_col_names[0]}' for URL "
            f"or '{ possible_image_path_col_names[0]}' for path in Team Files"
        )
    if len(col_names_validate) == 2:
        if not any(tag_name in possible_tag_col_names for tag_name in col_names_validate):
            raise ValueError(
                f"Invalid column name for tag: {col_names_validate[1]}. Please use:{ possible_tag_col_names[0]}"
            )


def validate_column_names(first_csv_row: dict):
    col_names_validate = [key.lower() for key in first_csv_row.keys()]
    col_names = [key for key in first_csv_row.keys()]
    check_column_names(col_names_validate)
    image_col_name = None
    tag_col_name = None
    for name in col_names:
        if name == "url":
            image_col_name = name
        if name == "path":
            image_col_name = name
        if name == "tag":
            tag_col_name = name

    return image_col_name, tag_col_name


def first_line_is_url(first_line: str):
    first_line = first_line.strip().lstrip('"').lstrip("'")
    markers = ["https://", "http://", "s3://", "azure://", "google://"]
    for marker in markers:
        if first_line.startswith(marker):
            return True
    return False


def first_line_is_path(first_line):
    first_line = first_line.strip().lstrip('"').lstrip("'")
    if first_line.startswith("/") or first_line.startswith("agent://"):
        return True
    return False


def handle_csv_header(csv_path):
    content = []
    updated = False
    with open(csv_path, "r") as file:
        # read the file, correct it and write it back
        content = file.readlines()
        if len(content) == 0:
            raise ValueError(f"File '{csv_path}' is empty")
        first_line = content[0]
        has_tags = False
        new_header = None
        for line in content:
            if DEFAULT_DELIMITER in line:
                has_tags = True
                break
        if first_line_is_url(first_line):
            logger.debug("Headless csv detected. Adding 'url' column name to the first line.")
            new_header = "url" + DEFAULT_DELIMITER + "tag\n" if has_tags else "url\n"
        elif first_line_is_path(first_line):
            logger.debug("Headless csv detected. Adding 'path' column name to the first line.")
            new_header = "path" + DEFAULT_DELIMITER + "tag\n" if has_tags else "path\n"
        if new_header is None:
            first_line = content[1]
            if first_line_is_url(first_line):
                new_header = "url" + DEFAULT_DELIMITER + "tag\n" if has_tags else "url\n"
            elif first_line_is_path(first_line):
                new_header = "path" + DEFAULT_DELIMITER + "tag\n" if has_tags else "path\n"
            if new_header is not None:
                content[0] = new_header
                updated = True
            else:
                raise ValueError(
                    f"Invalid header in csv file. "
                    f"Please use: 'url' for URLs or 'path' for paths in Team Files"
                )
        else:
            content.insert(0, new_header)
            updated = True

    if updated:
        silent_remove(csv_path)
        with open(csv_path, "w") as file:
            file.writelines(content)


def validate_and_collect_items(csv_path) -> Tuple[dict, ProjectMeta, list]:
    csv_table = {"columns": [], "data": []}

    if os.path.getsize(csv_path) == 0:
        raise ValueError(f"File '{csv_path}' is empty")
    handle_csv_header(csv_path)
    with open(csv_path, "r") as images_csv:
        try:
            reader = csv.DictReader(images_csv, delimiter=DEFAULT_DELIMITER)
            if len(reader.fieldnames) > 0 and INCORRECT_COLUMN_DELIMITER in reader.fieldnames[0]:
                raise ValueError(
                    f"Invalid column delimiter: '{ INCORRECT_COLUMN_DELIMITER}'. "
                    f"Please, use: '{ DEFAULT_DELIMITER}' - for column delimiter and '{ TAGS_DELIMITER}' - for tags delimiter"
                )
            reader = [row for row in reader]
        except Exception as e:
            raise Exception(f"Can't read csv file. {e}")
        stripped_reader = []
        if len(reader) == 0:
            raise ValueError(f"File '{csv_path}' is empty")
        logger.debug(f"Total rows in csv file: {len(reader)}")
        for row in reader:
            stripped_row = {
                k: v.strip() for k, v in row.items() if k is not None and type(v) == str
            }
            stripped_reader.append(stripped_row)
        csv_reader = stripped_reader
        row_to_validate = {}
        for row in stripped_reader:
            if len(row) > len(row_to_validate) and len(row) < 3:
                row_to_validate = row
        if len(row_to_validate) == 0:
            row_to_validate = stripped_reader[0]
        image_col_name, tag_col_name = validate_column_names(row_to_validate)

        if tag_col_name is not None:
            csv_table["columns"] = ["row", image_col_name, tag_col_name]
            for idx, row in enumerate(stripped_reader):
                csv_table["data"].append([idx + 1, row[image_col_name], row.get(tag_col_name, "")])
            total_tags = flat_tag_list(
                list(set([row.get(tag_col_name) for row in stripped_reader]))
            )
            project_meta = create_project_meta_from_csv_tags(total_tags)
        else:
            csv_table["columns"] = ["row", image_col_name]
            for idx, row in enumerate(stripped_reader):
                csv_table["data"].append([idx + 1, row[image_col_name]])
            total_tags = 0
            project_meta = ProjectMeta()

    return project_meta, csv_reader

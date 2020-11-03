# coding: utf-8

import pandas as pd
import urllib.parse

from supervisely_lib.api.labeling_job_api import LabelingJobApi
Status = LabelingJobApi.Status
import supervisely_lib.labeling_jobs.constants as constants
from supervisely_lib.api.module_api import ApiField


def total_desc():
    return "TOTAL", "the total number of jobs in current team"


def is_completed_desc():
    return "COMPLETED", "the number of completed jobs"


def is_completed(job_info):
    if job_info.status == str(Status.COMPLETED):
        return True


def is_stopped_desc():
    return "STOPPED", "the number of stopped jobs"


def is_stopped(job_info):
    if job_info.status == str(Status.STOPPED):
        return True


def is_not_started_desc():
    return "PENDING", "the number of jobs labeler haven't even opened yet (created but not started)"


def is_not_started(job_info):
    if job_info.status == str(Status.PENDING):
        return True


def total_items_count_desc():
    return "TOTAL", "the total number of items in all labeling jobs"


def total_items_count(job_info):
    return job_info.images_count


def labeled_items_count_desc():
    return "LABELED", "the total number of labeled items (labelers marked as \"finished\") in all labeling jobs"


# cnt images, that labeler marked as done
def labeled_items_count(job_info):
    if is_on_labeling(job_info):
        return job_info.finished_images_count
    else:
        return total_items_count(job_info)


def reviewed_items_count_desc():
    return "REVIEWED", "the total number of reviewed items (reviewers marked as \"accepted\" or \"rejected\") in all labeling jobs"


# cnt images, that reviewer accepted or rejected
def reviewed_items_count(job_info):
    return job_info.accepted_images_count + job_info.rejected_images_count


def accepted_items_count_desc():
    return "ACCEPTED", "the total number of accepted items (reviewers marked as \"accepted\") in all labeling jobs"


def accepted_items_count(job_info):
    return job_info.accepted_images_count


def rejected_items_count_desc():
    return "REJECTED", "the total number of rejected items (reviewers marked as \"rejected\") in all labeling jobs"


def rejected_items_count(job_info):
    return job_info.rejected_images_count


def is_on_labeling_desc():
    return 'LABELING IN PROGRESS', "the number of jobs with status IN_PROGRESS"


# labeling is in progress
def is_on_labeling(job_info):
    if job_info.status == str(Status.IN_PROGRESS):
        return True
    return False


def is_labeling_started_desc():
    return 'LABELING STARTED', "the number of jobs that are started by labeler and with at least one labeled item (marked \"done\" by labeler)"


# cnt jobs that are started by labeler and with at least one image that marked "done" by labeler
def is_labeling_started(job_info):
    if is_on_labeling(job_info) and labeled_items_count(job_info) != 0:
        return True
    return False


def is_on_review_desc():
    return 'ON REVIEW', "the number of jobs with status ON_REVIEW"


# cnt jobs with "on review" status
def is_on_review(job_info):
    if job_info.status == str(Status.ON_REVIEW):
        return True
    return False


def is_review_started_desc():
    return 'REVIEW STARTED', "the number of jobs that are started by reviewer - with at least one reviewed item (marked \"accepted\" or \"rejected\")"


# cnt jobs with at least one reviewed (accepted or rejected) item
def is_review_started(job_info):
    if is_on_review(job_info) and reviewed_items_count(job_info) != 0:
        return True
    return False


def is_zero_labeling_desc():
    return 'ZERO LABELED', "the number of jobs with status \"IN PROGRESS\" with zero labeled items"


def is_zero_reviewed_desc():
    return 'ZERO REVIEWED', "the number of jobs with status \"ON REVIEW\" with zero reviewed items"


def get_job_url(server_address, job):
    result = urllib.parse.urljoin(server_address, 'app/images/{}/{}/{}/{}?jobId={}'.format(job.team_id,
                                                                                          job.workspace_id,
                                                                                          job.project_id,
                                                                                          job.dataset_id,
                                                                                          job.id))

    return result


def jobs_stats(server_address, jobs, stats):
    col_job_id = []
    col_job_name = []  # link here
    col_job_status = []
    col_items_total = []
    col_items_labeled = []
    col_items_reviewed = []
    col_items_accepted = []
    col_items_rejected = []
    col_created_at = []

    for job, stat in zip(jobs, stats):
        col_job_id.append(job.id)
        col_job_name.append('<a href="{0}" rel="noopener noreferrer" target="_blank">{1}</a>'
                            .format(get_job_url(server_address, job), job.name))
        col_job_status.append(job.status)
        col_created_at.append(job.created_at)
        col_items_total.append(total_items_count(job))
        col_items_labeled.append(labeled_items_count(job))
        col_items_reviewed.append(reviewed_items_count(job))
        col_items_accepted.append(accepted_items_count(job))
        col_items_rejected.append(rejected_items_count(job))

    df = pd.DataFrame(list(zip(col_job_id,
                               col_job_name,
                               col_job_status,
                               col_items_total,
                               col_items_labeled,
                               col_items_reviewed,
                               col_items_accepted,
                               col_items_rejected,
                               col_created_at)),
                      columns=['ID', 'NAME', 'STATUS', 'TOTAL', 'LABELED', 'REVIEWED', 'ACCEPTED', 'REJECTED', 'CREATED_AT'])

    df['CREATED_AT'] = pd.to_datetime(df['CREATED_AT']).dt.strftime('%d/%m/%Y %H:%M')
    return df


def jobs_summary(jobs):
    count_total = len(jobs)

    count_completed = 0
    count_stopped = 0

    count_labeling_not_started = 0  # pending

    count_on_labeling = 0
    count_labeling_started = 0
    count_labeling_zero_done = 0

    count_on_review = 0
    count_review_started = 0
    count_review_zero_done = 0

    for job in jobs:
        if is_completed(job):
            count_completed += 1
        elif is_stopped(job):
            count_stopped += 1
        elif is_not_started(job):
            count_labeling_not_started += 1
        elif is_on_labeling(job):
            count_on_labeling += 1
            if is_labeling_started(job):
                count_labeling_started += 1
        elif is_on_review(job):
            count_on_review += 1
            if is_review_started(job):
                count_review_started += 1
        else:
            raise RuntimeError("Unhandled job status: {}".format(str(job)))

    count_labeling_zero_done = count_on_labeling - count_labeling_started
    count_review_zero_done = count_on_review - count_review_started

    names = []
    percentages = []
    descriptions = []

    table_rows_f = [total_desc, is_completed_desc, is_stopped_desc,
                    is_not_started_desc,
                    is_on_labeling_desc, is_labeling_started_desc, is_zero_labeling_desc,
                    is_on_review_desc, is_review_started_desc, is_zero_reviewed_desc]

    values = [count_total,
              count_completed,
              count_stopped,

              count_labeling_not_started,

              count_on_labeling,
              count_labeling_started,
              count_labeling_zero_done,

              count_on_review,
              count_review_started,
              count_review_zero_done,
              ]

    for v, func in zip(values, table_rows_f):
        percentages.append("{:.2f} %".format(v * 100 / count_total))
        name, desc = func()
        names.append(name)
        descriptions.append(desc)

    df = pd.DataFrame(list(zip(list(range(len(names))), names, values, percentages, descriptions)),
                      columns=['#', 'JOB STATUS', 'QUANTITY', 'PERCENTAGE', 'DESCRIPTION'])
    return df


def images_summary(jobs):
    count_total_items = 0
    count_labeled_items = 0
    count_reviewed_items = 0
    count_accepted_items = 0
    count_rejected_items = 0

    for job in jobs:
        count_total_items += total_items_count(job)
        count_labeled_items += labeled_items_count(job)
        count_reviewed_items += reviewed_items_count(job)
        count_accepted_items += accepted_items_count(job)
        count_rejected_items += rejected_items_count(job)

    values_items = [count_total_items,
                    count_labeled_items,
                    count_reviewed_items,
                    count_accepted_items,
                    count_rejected_items]

    items_f = [total_items_count_desc,
               labeled_items_count_desc,
               reviewed_items_count_desc,
               accepted_items_count_desc,
               rejected_items_count_desc]

    names_items = []
    percentages_items = []
    descriptions_items = []

    for v, func in zip(values_items, items_f):
        percentages_items.append("{:.2f} %".format(v * 100 / count_total_items))
        name, desc = func()
        names_items.append(name)
        descriptions_items.append(desc)

    df = pd.DataFrame(list(zip(list(range(len(names_items))), names_items, values_items, percentages_items, descriptions_items)),
                      columns=['#', 'ITEM STATUS', 'QUANTITY', 'PERCENTAGE', 'DESCRIPTION'])
    return df


def classes_summary(stats):
    class_id_stats = {}
    for stat in stats:
        for class_stat in stat[constants.CLASSES_STATS]:
            class_id = class_stat[ApiField.ID]
            class_name = class_stat[ApiField.NAME]
            class_shape = class_stat[ApiField.SHAPE]
            image_count = class_stat[ApiField.IMAGES_COUNT]
            object_count = class_stat[ApiField.LABELS_COUNT]
            class_color = class_stat[ApiField.COLOR]

            if class_id not in class_id_stats:
                class_id_stats[class_id] = {ApiField.NAME: class_name, ApiField.SHAPE: class_shape,
                                            ApiField.COLOR: class_color,
                                            ApiField.IMAGES_COUNT: 0, ApiField.LABELS_COUNT: 0}

            class_id_stats[class_id][ApiField.IMAGES_COUNT] += image_count
            class_id_stats[class_id][ApiField.LABELS_COUNT] += object_count

    col_name = []
    col_shape = []
    col_image_count = []
    col_object_count = []
    for class_id, value in class_id_stats.items():
        col_name.append('<b style="display: inline-block; border-radius: 50%; background: {}; width: 8px; height: 8px"></b> {}'
                        .format(value[ApiField.COLOR], value[ApiField.NAME]))
        col_shape.append(value[ApiField.SHAPE])
        col_image_count.append(value[ApiField.IMAGES_COUNT])
        col_object_count.append(value[ApiField.LABELS_COUNT])

    df = pd.DataFrame(list(zip(list(range(len(col_name))), col_name, col_shape, col_image_count, col_object_count)),
                      columns=['#', 'CLASS', 'SHAPE', 'IMAGES COUNT', 'OBJECTS COUNT'])
    return df


def tags_summary(stats):
    tag_id_stats = {}
    for stat in stats:
        for tag_stat in stat["job"][constants.TAGS_STATS]:
            if "parentId" in tag_stat:
                # @TODO: consider parentId (for oneOf cases)
                continue
            tag_id = tag_stat[ApiField.ID]
            tag_name = tag_stat[ApiField.NAME]
            image_count = tag_stat[ApiField.IMAGES]
            object_count = tag_stat[ApiField.FIGURES]
            class_color = tag_stat[ApiField.COLOR]

            if tag_id not in tag_id_stats:
                tag_id_stats[tag_id] = {ApiField.NAME: tag_name, ApiField.COLOR: class_color,
                                        ApiField.IMAGES_COUNT: 0, ApiField.LABELS_COUNT: 0}

            tag_id_stats[tag_id][ApiField.IMAGES_COUNT] += image_count
            tag_id_stats[tag_id][ApiField.LABELS_COUNT] += object_count

    col_name = []
    col_image_count = []
    col_object_count = []
    for tag_id, value in tag_id_stats.items():
        col_name.append('<b style="display: inline-block; border-radius: 50%; background: {}; width: 8px; height: 8px"></b> {}'
                        .format(value[ApiField.COLOR], value[ApiField.NAME]))
        col_image_count.append(value[ApiField.IMAGES_COUNT])
        col_object_count.append(value[ApiField.LABELS_COUNT])

    df = pd.DataFrame(list(zip(list(range(len(col_name))), col_name, col_image_count, col_object_count)),
                      columns=['#', 'TAG', 'IMAGES COUNT', 'OBJECTS COUNT'])
    return df
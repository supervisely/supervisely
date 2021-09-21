# coding: utf-8
import time
import pandas as pd
from supervisely_lib.collection.str_enum import StrEnum
from supervisely_lib.api.module_api import ApiField, ModuleApi, RemoveableModuleApi, ModuleWithStatus, \
                                           WaitingTimeExceeded


class LabelingJobApi(RemoveableModuleApi, ModuleWithStatus):
    class Status(StrEnum):
        PENDING = 'pending'
        IN_PROGRESS = "in_progress"
        ON_REVIEW = "on_review"
        COMPLETED = "completed"
        STOPPED = 'stopped'

    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.NAME,
                ApiField.README,
                ApiField.DESCRIPTION,

                ApiField.TEAM_ID,
                ApiField.WORKSPACE_ID,
                ApiField.WORKSPACE_NAME,
                ApiField.PROJECT_ID,
                ApiField.PROJECT_NAME,
                ApiField.DATASET_ID,
                ApiField.DATASET_NAME,

                ApiField.CREATED_BY_ID,
                ApiField.CREATED_BY_LOGIN,
                ApiField.ASSIGNED_TO_ID,
                ApiField.ASSIGNED_TO_LOGIN,
                ApiField.REVIEWER_ID,
                ApiField.REVIEWER_LOGIN,

                ApiField.CREATED_AT,
                ApiField.STARTED_AT,
                ApiField.FINISHED_AT,
                ApiField.STATUS,
                ApiField.DISABLED,
                ApiField.IMAGES_COUNT,
                ApiField.FINISHED_IMAGES_COUNT,
                ApiField.REJECTED_IMAGES_COUNT,
                ApiField.ACCEPTED_IMAGES_COUNT,
                ApiField.PROGRESS_IMAGES_COUNT,

                ApiField.CLASSES_TO_LABEL,
                ApiField.TAGS_TO_LABEL,
                ApiField.IMAGES_RANGE,
                ApiField.OBJECTS_LIMIT_PER_IMAGE,
                ApiField.TAGS_LIMIT_PER_IMAGE,

                ApiField.FILTER_IMAGES_BY_TAGS,
                ApiField.INCLUDE_IMAGES_WITH_TAGS,
                ApiField.EXCLUDE_IMAGES_WITH_TAGS,

                ApiField.ENTITIES,
                ]

    @staticmethod
    def info_tuple_name():
        return 'LabelingJobInfo'

    def __init__(self, api):
        ModuleApi.__init__(self, api)

    def _convert_json_info(self, info: dict, skip_missing=True):
        if info is None:
            return None
        else:
            field_values = []
            for field_name in self.info_sequence():
                if field_name in [ApiField.INCLUDE_IMAGES_WITH_TAGS, ApiField.EXCLUDE_IMAGES_WITH_TAGS]:
                    continue
                value = None
                if type(field_name) is str:
                    if skip_missing is True:
                        value = info.get(field_name, None)
                    else:
                        value = info[field_name]
                elif type(field_name) is tuple:
                    for sub_name in field_name[0]:
                        if value is None:
                            if skip_missing is True:
                                value = info.get(sub_name, None)
                            else:
                                value = info[sub_name]
                        else:
                            value = value[sub_name]
                else:
                    raise RuntimeError('Can not parse field {!r}'.format(field_name))

                if field_name == ApiField.FILTER_IMAGES_BY_TAGS:
                    field_values.append(value)
                    include_images_with_tags = []
                    exclude_images_with_tags = []
                    for fv in value:
                        key = ApiField.NAME
                        if key not in fv:
                            key = 'title'
                        if fv['positive'] is True:
                            include_images_with_tags.append(fv[key])
                        else:
                            exclude_images_with_tags.append(fv[key])
                    field_values.append(include_images_with_tags)
                    field_values.append(exclude_images_with_tags)
                    continue
                elif field_name == ApiField.CLASSES_TO_LABEL or field_name == ApiField.TAGS_TO_LABEL:
                    value = []
                    for fv in value:
                        key = ApiField.NAME
                        if ApiField.NAME not in fv:
                            key = 'title'
                        value.append(fv[key])
                elif field_name == ApiField.IMAGES_RANGE:
                    value = (value['start'], value['end'])

                field_values.append(value)
            return self.InfoType(*field_values)

    def create(self,
               name,
               dataset_id,
               user_ids,
               readme=None,
               description=None,
               classes_to_label=None,
               objects_limit_per_image=None,
               tags_to_label=None,
               tags_limit_per_image=None,
               include_images_with_tags=None,
               exclude_images_with_tags=None,
               images_range=None,
               reviewer_id=None):
        '''
        Create labeling job by given user in given dataset
        :param name: str
        :param dataset_id: int
        :param user_ids: list of integers
        :param readme: str
        :param description: str
        :param classes_to_label: list
        :param objects_limit_per_image: int
        :param tags_to_label: list
        :param tags_limit_per_image: int
        :param include_images_with_tags: bool
        :param exclude_images_with_tags: bool
        :param images_range: list
        :return: list
        '''
        if classes_to_label is None:
            classes_to_label = []
        if tags_to_label is None:
            tags_to_label = []

        filter_images_by_tags = []
        if include_images_with_tags is not None:
            for tag_name in include_images_with_tags:
                filter_images_by_tags.append({'name': tag_name, 'positive': True})

        if exclude_images_with_tags is not None:
            for tag_name in exclude_images_with_tags:
                filter_images_by_tags.append({'name': tag_name, 'positive': False})

        if objects_limit_per_image is None:
            objects_limit_per_image = 0

        if tags_limit_per_image is None:
            tags_limit_per_image = 0

        data = {ApiField.NAME: name,
                ApiField.DATASET_ID: dataset_id,
                ApiField.USER_IDS: user_ids,
                #ApiField.DESCRIPTION: description,
                ApiField.META: {
                     'classes': classes_to_label,
                     'projectTags': tags_to_label,
                     'imageTags': filter_images_by_tags,
                     'imageFiguresLimit': objects_limit_per_image,
                     'imageTagsLimit': tags_limit_per_image,}
                }

        if readme is not None:
            data[ApiField.README] = str(readme)

        if description is not None:
            data[ApiField.DESCRIPTION] = str(description)

        if images_range is not None:
            if len(images_range) != 2:
                raise RuntimeError('images_range has to contain 2 elements (start, end)')
            images_range = {'start': images_range[0], 'end': images_range[1]}
            data[ApiField.META]['range'] = images_range

        if reviewer_id is not None:
            data[ApiField.REVIEWER_ID] = reviewer_id

        response = self._api.post('jobs.add', data)
        # created_jobs_json = response.json()

        created_jobs = []
        for job in response.json():
            created_jobs.append(self.get_info_by_id(job[ApiField.ID]))
        return created_jobs

    def get_list(self, team_id, created_by_id=None, assigned_to_id=None, project_id=None, dataset_id=None,
                 show_disabled=False):
        '''
        Get all labeling that were created by given user and were assigned to given assigned user in given team
        :param team_id: int
        :param created_by_id: int
        :param assigned_to_id:
        :param project_id:
        :param dataset_id:
        :return: list
        '''
        filters = []
        if created_by_id is not None:
            filters.append({"field": ApiField.CREATED_BY_ID[0][0], "operator": "=", "value": created_by_id})
        if assigned_to_id is not None:
            filters.append({"field": ApiField.ASSIGNED_TO_ID[0][0], "operator": "=", "value": assigned_to_id})
        if project_id is not None:
            filters.append({"field": ApiField.PROJECT_ID, "operator": "=", "value": project_id})
        if dataset_id is not None:
            filters.append({"field": ApiField.DATASET_ID, "operator": "=", "value": dataset_id})
        return self.get_list_all_pages('jobs.list', {ApiField.TEAM_ID: team_id,
                                                     "showDisabled": show_disabled,
                                                     ApiField.FILTER: filters})

    def stop(self, id):
        '''
        :param id: int
        :return: stop labeling job  job will be unavailable for labeler with given id
        '''
        self._api.post('jobs.stop', {ApiField.ID: id})

    def get_info_by_id(self, id):
        '''
        :param id: int
        :return: labeling job with given id
        '''
        return self._get_info_by_id(id, 'jobs.info')

    def archive(self, id):
        '''
        Archive labeling job with given id
        :param id: int
        '''
        self._api.post('jobs.archive', {ApiField.ID: id})

    def get_status(self, id):
        '''
        :param id: int
        :return: status of labeling job with given id
        '''
        status_str = self.get_info_by_id(id).status
        return self.Status(status_str)

    def raise_for_status(self, status):
        #there is no ERROR status for labeling job
        pass

    def wait(self, id, target_status, wait_attempts=None, wait_attempt_timeout_sec=None):
        '''
        Awaiting achievement by given labeling job of a given status
        :param id: int
        :param target_status: status of labeling job we expect to destinate
        :param wait_attempts: int
        :param wait_attempt_timeout_sec: int (raise error if waiting time exceeded)
        :return: bool
        '''
        wait_attempts = wait_attempts or self.MAX_WAIT_ATTEMPTS
        effective_wait_timeout = wait_attempt_timeout_sec or self.WAIT_ATTEMPT_TIMEOUT_SEC
        for attempt in range(wait_attempts):
            status = self.get_status(id)
            self.raise_for_status(status)
            if status is target_status:
                return
            time.sleep(effective_wait_timeout)
        raise WaitingTimeExceeded('Waiting time exceeded')

    def get_stats(self, id):
        '''
        :param id: int
        :return: status of given labeling job
        '''
        response = self._api.post('jobs.stats', {ApiField.ID: id})
        return response.json()

    def get_activity(self, team_id, job_id, progress_cb=None):
        '''
        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param job_id: Labeling Job ID in Supervisely.
        :type team_id: int
        :return: pandas dataframe
        '''
        activity = self._api.team.get_activity(team_id, filter_job_id=job_id, progress_cb=progress_cb)
        df = pd.DataFrame(activity)
        return df

import copy

import fastapi

from supervisely.app.widgets import Widget


class ProjectSelector(Widget):
    # @TODO: add Routes project changes events
    # class Routes:
    #     def __init__(self,
    #                  app: fastapi.FastAPI,
    #                  cell_clicked_cb: object = None):
    #         self.app = app
    #         self.routes = {'cell_clicked_cb': cell_clicked_cb}

    def __init__(self,
                 team_id: int = None,
                 team_is_selectable: bool = True,
                 datasets_is_selectable: bool = True,
                 widget_id: str = None):

        self._team_id = team_id

        self._is_selectable = {
            'team': team_is_selectable,
            'datasets': datasets_is_selectable
        }

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            'selectable': copy.copy(self._is_selectable)
        }

    def get_json_state(self):
        return {
            'teamId': self._team_id,
            'workspaceId': None,
            'projectId': None,
            'allDatasets': True,
            'datasetsIds': []
        }

    def get_selected_team_id(self, state):
        return state[self.widget_id]['teamId']

    def get_selected_workspace_id(self, state):
        return state[self.widget_id]['workspaceId']

    def get_selected_project_id(self, state):
        return state[self.widget_id]['projectId']

    def get_selected_datasets(self, state):
        datasets = []
        if self._is_selectable['datasets']:
            datasets = state[self.widget_id]['datasetsIds']
        return datasets


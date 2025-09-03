import os
import typing
from os import PathLike

import jinja2
from fastapi.templating import Jinja2Templates as _fastapi_Jinja2Templates
from starlette.background import BackgroundTask
from starlette.templating import _TemplateResponse as _TemplateResponse

from supervisely.app.singleton import Singleton
from supervisely.app.widgets_context import JinjaWidgets

# https://github.com/supervisely/js-bundle
js_bundle_version = "2.2.2"

# https://github.com/supervisely-ecosystem/supervisely-app-frontend-js
js_frontend_version = "v0.0.56"


pyodide_version = "v0.25.0"


class Jinja2Templates(_fastapi_Jinja2Templates, metaclass=Singleton):
    def __init__(self, directory: typing.Union[str, PathLike] = "templates") -> None:
        super().__init__(directory)

    def _create_env(self, directory: typing.Union[str, PathLike]) -> "jinja2.Environment":
        env_fastapi = super()._create_env(directory)
        env_sly = jinja2.Environment(
            loader=env_fastapi.loader,
            autoescape=True,
            variable_start_string="{{{",
            variable_end_string="}}}",
        )
        try:
            env_sly.globals["url_for"] = env_fastapi.globals["url_for"]
        except:
            # for fastapi version==0.108.0
            pass
        return env_sly

    def TemplateResponse(
        self,
        name: str,
        context: dict,
        status_code: int = 200,
        headers: dict = None,
        media_type: str = None,
        background: BackgroundTask = None,
    ) -> _TemplateResponse:
        from supervisely.app.fastapi.subapp import get_name_from_env

        context_with_widgets = {
            **context,
            **JinjaWidgets().context,
            "js_bundle_version": js_bundle_version,
            "js_frontend_version": js_frontend_version,
            "app_name": get_name_from_env(default="Supervisely App"),
        }

        try:
            request = context["request"]
            return super().TemplateResponse(  # pylint: disable=no-member too-many-function-args
                request, name, context_with_widgets, status_code, headers, media_type, background
            )
        except:
            # for fastapi version<0.108.0
            return super().TemplateResponse(
                name, context_with_widgets, status_code, headers, media_type, background
            )

    def render(self, name: str, context: dict) -> str:
        from supervisely.app.fastapi.subapp import get_name_from_env

        context_with_widgets = {
            **context,
            **JinjaWidgets().context,
            "js_bundle_version": js_bundle_version,
            "js_frontend_version": js_frontend_version,
            "pyodide_version": pyodide_version,
            "app_name": get_name_from_env(default="Supervisely App"),
        }
        template: jinja2.Template = self.get_template(name)
        return template.render(context_with_widgets)

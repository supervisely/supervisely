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
js_bundle_version = "2.2.5"

# https://github.com/supervisely-ecosystem/supervisely-app-frontend-js
js_frontend_version = "v0.0.56"


pyodide_version = "v0.25.0"


def _call_fastapi_create_env(
    create_env: typing.Callable[..., "jinja2.Environment"],
    instance: "Jinja2Templates",
    directory: typing.Union[str, PathLike],
) -> "jinja2.Environment":
    return create_env(instance, directory)


class Jinja2Templates(_fastapi_Jinja2Templates, metaclass=Singleton):
    """FastAPI Jinja2 templates with Supervisely widget context and custom variable delimiters ({{{ }}})."""

    def __init__(self, directory: typing.Union[str, PathLike] = "templates") -> None:
        """
        :param directory: Path to templates directory.
        :type directory: typing.Union[str, PathLike]
        """
        if hasattr(_fastapi_Jinja2Templates, "_create_env"):
            super().__init__(directory)
        else:
            loader = jinja2.FileSystemLoader(directory)
            super().__init__(env=self._create_sly_env(loader))

    @staticmethod
    def _create_sly_env(loader: "jinja2.BaseLoader") -> "jinja2.Environment":
        env_sly = jinja2.Environment(
            loader=loader,
            autoescape=True,
            variable_start_string="{{{",
            variable_end_string="}}}",
        )
        return env_sly

    def _create_env(self, directory: typing.Union[str, PathLike]) -> "jinja2.Environment":
        create_env = getattr(_fastapi_Jinja2Templates, "_create_env", None)
        if not callable(create_env):
            loader = jinja2.FileSystemLoader(directory)
            return self._create_sly_env(loader)

        env_fastapi = _call_fastapi_create_env(create_env, self, directory)
        env_sly = self._create_sly_env(env_fastapi.loader)
        if "url_for" in env_fastapi.globals:
            env_sly.globals["url_for"] = env_fastapi.globals["url_for"]
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

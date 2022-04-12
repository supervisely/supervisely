import typing
from os import PathLike
import jinja2
from fastapi.templating import Jinja2Templates as _fastapi_Jinja2Templates
from starlette.templating import _TemplateResponse as _TemplateResponse
from starlette.background import BackgroundTask
from supervisely.app.singleton import Singleton


class Jinja2Templates(_fastapi_Jinja2Templates, metaclass=Singleton):
    def __init__(self, directory: typing.Union[str, PathLike] = "templates") -> None:
        super().__init__(directory)
        self.context_widgets = {}

    def _create_env(
        self, directory: typing.Union[str, PathLike]
    ) -> "jinja2.Environment":
        env_fastapi = super()._create_env(directory)
        env_sly = jinja2.Environment(
            loader=env_fastapi.loader,
            autoescape=True,
            variable_start_string="{{{",
            variable_end_string="}}}",
        )
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
        context_with_widgets = {**context, **self.context_widgets}

        return super().TemplateResponse(
            name, context_with_widgets, status_code, headers, media_type, background
        )

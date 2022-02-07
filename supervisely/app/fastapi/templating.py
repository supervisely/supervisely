import typing
from os import PathLike
import jinja2
from fastapi.templating import Jinja2Templates as _fastapi_Jinja2Templates


class Jinja2Templates(_fastapi_Jinja2Templates):
    def _create_env(
        self, directory: typing.Union[str, PathLike]
    ) -> "jinja2.Environment":
        env_fastapi = super()._create_env(directory)
        env_sly = jinja2.Environment(
            loader=env_fastapi.loader, 
            autoescape=True,
            variable_start_string='{{{',
            variable_end_string='}}}',
        )
        env_sly.globals["url_for"] = env_fastapi.globals["url_for"]
        return env_sly
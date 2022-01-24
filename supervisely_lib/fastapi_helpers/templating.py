import typing
from os import PathLike

import jinja2
from fastapi.templating import pass_context
from fastapi.templating import Jinja2Templates as FastAPIJinja2Templates


class Jinja2Templates(FastAPIJinja2Templates):
    def _create_env(
        self, directory: typing.Union[str, PathLike]
    ) -> "jinja2.Environment":
        @pass_context
        def url_for(context: dict, name: str, **path_params: typing.Any) -> str:
            request = context["request"]
            return request.url_for(name, **path_params)

        loader = jinja2.FileSystemLoader(directory)
        env = jinja2.Environment(
            loader=loader, 
            autoescape=True,
            variable_start_string='{{{',
            variable_end_string='}}}',

        )
        env.globals["url_for"] = url_for
        return env
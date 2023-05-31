from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from typing import Dict


class Docstring(Widget):
    def __init__(
        self,
        content: str = "",
        is_html: bool = False,
        widget_id: str = None,
    ):
        self._is_html = is_html

        if self._is_html is True:
            self._content = content
        else:
            self._content = self._convert_docstring_to_html(content)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {"content": self._content}

    def get_json_state(self) -> Dict:
        return {}

    def set_content(
        self,
        content: str,
        is_html: bool = False,
    ):
        if is_html is False:
            content = self._convert_docstring_to_html(content)
        self._content = content
        DataJson()[self.widget_id]["content"] = self._content
        DataJson().send_changes()

    def clear(self):
        self._content = ""
        DataJson()[self.widget_id]["content"] = self._content
        DataJson().send_changes()

    def _convert_docstring_to_html(self, docstring: str) -> str:
        import docutils.core
        from docutils.parsers.rst import roles

        parts = docstring.split('\n"""')
        if len(parts) > 1:
            docstring = parts[1].strip('"\n')

        roles.register_canonical_role("class", roles.generic_custom_role)

        html_body = docutils.core.publish_parts(
            docstring,
            writer_name="html",
            settings_overrides={
                "initial_header_level": 2,
                "input_encoding": "unicode",
                "exit_status_level": 2,
                "pep_references": None,
            },
        )
        return html_body["body"]

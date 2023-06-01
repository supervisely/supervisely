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

        if self._is_html is True or content == "":
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

    def _convert_docstring_to_html(self, docstring):
        import os
        from bs4 import BeautifulSoup
        from supervisely.io.fs import remove_dir

        # Sphinx conf.py
        conf = """import os
import sys

sys.path.insert(0, os.path.abspath("my_file"))

project = "Python"
copyright = ""
author = ""
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

html_show_sourcelink = False

html_theme = "sphinx_rtd_theme"
html_theme_options = {
'analytics_anonymize_ip': True,
'logo_only': True,
'display_version': False,
'prev_next_buttons_location': 'None',
'style_external_links': False,
'vcs_pageview_mode': '',
'style_nav_header_background': 'white',
'collapse_navigation': True,
'sticky_navigation': False,
'navigation_depth': 0,
'includehidden': False,
'titles_only': True
}

"""
        py_code = '''
class InsertDocString:"""
{}
"""
'''
        text = py_code.format(docstring)

        with open("conf.py", "w") as config:
            config.write(conf)

        # Text variable
        os.mkdir("my_file")
        with open("my_file/my_script.py", "w") as my_script:
            my_script.write(text)

        # rst
        rst_conf = """modules
=======

.. toctree::
:maxdepth: 4

modules
"""
        with open("index.rst", "w") as rst:
            rst.write(rst_conf)

        # File structure and cleaning
        os.system("sphinx-apidoc -o .  my_file")
        os.system("sphinx-build -b html . _build")

        with open("_build/my_script.html") as f:
            html_content = f.read()
            soup = BeautifulSoup(html_content, "html.parser")
            result = soup.find("dd")

            if result is None:
                raise Exception("Not found")

            result.find("p").decompose()

        os.remove("index.rst")
        os.remove("modules.rst")
        os.remove("my_script.rst")
        os.remove("conf.py")
        remove_dir("my_file")
        remove_dir("_build")

        return str(result)

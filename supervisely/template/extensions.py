from jinja2.ext import Extension
import markdown
import textwrap
from jinja2.nodes import CallBlock

class MarkdownExtension(Extension):
    EXTENSIONS = [
        "admonition",
        "attr_list",
        "smarty",
        "tables",
        "pymdownx.betterem",
        "pymdownx.caret",
        "pymdownx.details",
        "pymdownx.emoji",
        "pymdownx.keys",
        "pymdownx.magiclink",
        "pymdownx.mark",
        "pymdownx.smartsymbols",
        "pymdownx.superfences",
        "pymdownx.tabbed",
        "pymdownx.tasklist",
        "pymdownx.tilde",
        "pymdownx.highlight",
    ]

    EXTENSION_CONFIGS = {
        'pymdownx.highlight': {
            'use_pygments': True,
            'noclasses': True,
            # "linenums": True, 
            # "linenums_style": "pymdownx-inline",
        },
    }
    tags = set(["markdown"])

    def __init__(self, environment):
        super(MarkdownExtension, self).__init__(environment)
        environment.extend(
            markdowner=markdown.Markdown(extensions=self.EXTENSIONS, extension_configs=self.EXTENSION_CONFIGS)
        )

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(
            ["name:endmarkdown"],
            drop_needle=True
        )
        return CallBlock(
            self.call_method("_render_markdown"),
            [],
            [],
            body
        ).set_lineno(lineno)

    def _render_markdown(self, caller):
        text = caller()
        text = self._dedent(text)
        return self.environment.markdowner.convert(text)

    def _dedent(self, text):
        return textwrap.dedent(text.strip("\n"))


class AutoSidebarExtension(Extension):
    tags = {'autosidebar'}
    
    def parse(self, parser):
        lineno = next(parser.stream).lineno
        return CallBlock(
            self.call_method('_placeholder'),
            [], [], []
        ).set_lineno(lineno)
    
    def _placeholder(self, caller):
        return "<!--AUTOSIDEBAR_PLACEHOLDER-->"

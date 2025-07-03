import json
import re
import textwrap

try:
    import markdown  # type: ignore
except ImportError:  # pragma: no cover
    markdown = None  # type: ignore
from jinja2.ext import Extension
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
        "pymdownx.highlight": {
            "use_pygments": True,
            "noclasses": True,
            # "linenums": True,
            # "linenums_style": "pymdownx-inline",
        },
    }
    tags = set(["markdown"])

    def __init__(self, environment):
        super(MarkdownExtension, self).__init__(environment)
        if markdown is None:
            raise ImportError("markdown library is required for MarkdownExtension but is not installed.")

        environment.extend(
            markdowner=markdown.Markdown(
                extensions=self.EXTENSIONS, extension_configs=self.EXTENSION_CONFIGS
            )
        )

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endmarkdown"], drop_needle=True)
        return CallBlock(self.call_method("_render_markdown"), [], [], body).set_lineno(lineno)

    def _render_markdown(self, caller):
        text = caller()
        text = self._dedent(text)
        return self.environment.markdowner.convert(text)

    def _dedent(self, text):
        return textwrap.dedent(text.strip("\n"))


class AutoSidebarExtension(Extension):
    tags = {"autosidebar"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        return CallBlock(self.call_method("_placeholder"), [], [], []).set_lineno(lineno)

    def _placeholder(self, caller):
        return "<!--AUTOSIDEBAR_PLACEHOLDER-->\n"


def _slugify(text: str) -> str:
    """Simple slugify helper (kebab-case)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "tab"


class TabExtension(Extension):
    """
    Internal helper extension. Each `{% tab title="X" %} ... {% endtab %}` inside a
    `{% tabs %} ... {% endtabs %}` block will register its content in a temporary
    buffer on the Jinja2 `Environment` (``_tab_buffer``).
    The buffer is consumed by the outer `TabsExtension` which renders the final
    HTML with `<sly-iw-tabs>` component.
    """

    tags = {"tab"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        # Expect: tab title="Some Title"
        parser.stream.expect("name:title")
        parser.stream.expect("assign")  # '=' symbol
        title_expr = parser.parse_expression()
        body = parser.parse_statements(["name:endtab"], drop_needle=True)
        return CallBlock(self.call_method("_render_tab", [title_expr]), [], [], body).set_lineno(
            lineno
        )

    def _render_tab(self, title: str, caller):
        raw_content = caller()
        # content_html = self.environment.markdowner.convert(raw_content)

        buf = getattr(self.environment, "_tab_buffer", None)
        if buf is not None:
            buf.append((title, raw_content))
            # buf.append((title, content_html))
        return ""


class TabsExtension(Extension):
    """
    Implements a GitBook-style tabs syntax:

    ```
    {% tabs %}
    {% tab title="Windows" %}
    Windows content
    {% endtab %}
    {% tab title="Linux" %}
    Linux content
    {% endtab %}
    {% endtabs %}
    ```
    """

    tags = {"tabs"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endtabs"], drop_needle=True)
        return CallBlock(self.call_method("_render_tabs"), [], [], body).set_lineno(lineno)

    def _render_tabs(self, caller):
        # Prepare a buffer for nested tab blocks
        self.environment._tab_buffer = []
        caller()  # Render inner content – `TabExtension` will fill the buffer
        tabs_data = self.environment._tab_buffer
        # Clean up
        delattr(self.environment, "_tab_buffer")

        if not tabs_data:
            # Nothing collected – return original caller() content
            return ""

        # Build tab definitions
        tab_defs = []
        rendered_templates = []
        for title, content in tabs_data:
            slug = _slugify(title)
            tab_defs.append({"name": slug, "title": title})
            rendered_templates.append(f"<template #{slug}>\n{content}\n</template>")

        tabs_json = json.dumps(tab_defs)
        html_parts = [
            "<div>",
            f"<sly-iw-tabs :tabs='{tabs_json}' :defaultIndex='0' :command='command' :data='data'>",
            *rendered_templates,
            "</sly-iw-tabs>",
            "</div>",
        ]
        return "\n".join(html_parts)

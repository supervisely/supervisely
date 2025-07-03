import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from jinja2 import Environment, FileSystemLoader

from supervisely import logger
from supervisely.template.extensions import (
    AutoSidebarExtension,
    MarkdownExtension,
    TabExtension,
    TabsExtension,
)


class TemplateRenderer:

    def __init__(
        self,
        generate_sidebar: bool = False,
        jinja_options: Optional[Dict[str, Any]] = None,
        jinja_extensions: Optional[list] = None,
        add_header_ids: bool = True,
    ):
        """
        Initializes template renderer with specified parameters.

        :param jinja_extensions: List of Jinja2 extensions to use. By default includes MarkdownExtension.
        :type jinja_extensions: Optional[list]
        :param jinja_options: Additional options for configuring Jinja2 environment.
        :type jinja_options: Optional[Dict[str, Any]]
        """
        if jinja_options is None:
            jinja_options = {
                "autoescape": False,
                "trim_blocks": True,
                "lstrip_blocks": True,
            }

        if jinja_extensions is None:
            jinja_extensions = [MarkdownExtension, TabsExtension, TabExtension]
        if generate_sidebar:
            jinja_extensions.append(AutoSidebarExtension)

        self.jinja_extensions = jinja_extensions
        self.jinja_options = jinja_options
        self.add_header_ids = add_header_ids
        self.environment = None

    def render(
        self,
        template_path: str,
        context: dict,
    ) -> str:
        """
        Renders a template with the provided context.

        :param template_path: Path to the markdown template file relative to template dir.
        :type template_path: str
        :param context: Dictionary with data for template rendering.
        :type context: Dict[str, Any]

        :returns: Result of template rendering as a string.
        :rtype: str
        """
        # Render
        p = Path(template_path)
        directory = p.parent.absolute()
        filename = p.name
        loader = FileSystemLoader(directory)
        env_options = {**self.jinja_options, "extensions": self.jinja_extensions}
        environment = Environment(loader=loader, **env_options)
        self.environment = environment
        template = environment.get_template(filename)
        html = template.render(**context)

        # Generate sidebar if AutoSidebarExtension is used
        if AutoSidebarExtension in self.jinja_extensions:
            html = self._generate_autosidebar(html)
        elif self.add_header_ids:
            html = self._add_header_ids(html)

        return html

    def _add_header_ids(self, content_html: str) -> str:
        """
        Add IDs to h2 and h3 header tags for table of contents generation.
        
        Args:
            content_html: HTML content with h2 and h3 headers
            
        Returns:
            HTML content with IDs added to headers
        """
        def clean_title_for_id(title: str) -> str:
            """Convert header title to a clean ID format"""
            # Remove HTML tags if any
            title = re.sub(r'<[^>]+>', '', title)
            # Keep only alphanumeric characters, spaces, and hyphens
            title = re.sub(r'[^\w\s-]', '', title)
            # Replace spaces with hyphens and convert to lowercase
            title = title.strip().lower().replace(' ', '-')
            # Remove multiple consecutive hyphens
            title = re.sub(r'-+', '-', title)
            # Remove leading/trailing hyphens
            title = title.strip('-')
            return title
        
        def replace_header(match):
            """Replace header with version that includes ID"""
            level = match.group(1)  # Header level (2 or 3)
            title = match.group(2).strip()  # Header title text
            
            # Generate clean ID
            clean_id = clean_title_for_id(title)
            section_id = f"{clean_id}"
            
            # Check for potential duplicate IDs (basic check)
            if section_id in used_ids:
                logger.debug(f"Duplicate header ID detected: '{section_id}' for title '{title}'")
                section_id += "-2"  # TODO: Improve duplicate handling logic
            used_ids.add(section_id)
            
            # Return header with ID attribute
            return f'<h{level} id="{section_id}">{title}</h{level}>'
        
        # Track used IDs for duplicate detection
        used_ids = set()
        
        # Pattern to match h2 and h3 headers
        header_pattern = r"<h([2-3])>(.*?)</h\1>"
        
        # Replace all matching headers with versions that include IDs
        updated_html = re.sub(header_pattern, replace_header, content_html, flags=re.IGNORECASE)
        
        return updated_html

    def _generate_autosidebar(self, content_html: str):
        # Extract h2 headers and generate ids
        h2_pattern = r"<h2>(.*?)</h2>"
        headers = []
        navigation = []
        lines = content_html.split("\n")
        for i, line in enumerate(lines):
            match = re.search(h2_pattern, line)
            if match:
                title = match.group(1).strip()
                section_id = f"{title.lower().replace(' ', '-')}-markdown-section"
                headers.append({"index": i, "title": title, "id": section_id})
                navigation.append({"id": section_id, "title": title})

        # Add ids to h2 headers
        new_lines = []
        first_header_index = headers[0]["index"]
        new_lines.extend(lines[:first_header_index])
        for i, header in enumerate(headers):
            new_lines.append(f'<div id="{header["id"]}" class="section">')
            new_lines.append(lines[header["index"]])
            if i < len(headers) - 1:
                next_header_index = headers[i + 1]["index"]
                new_lines.extend(lines[header["index"] + 1 : next_header_index])
                new_lines.append("</div>")
            else:
                closing_found = False
                for j in range(header["index"] + 1, len(lines)):
                    if lines[j].strip().startswith("</sly-iw-sidebar>"):
                        new_lines.append("</div>")
                        new_lines.extend(lines[j:])
                        closing_found = True
                        break
                    else:
                        new_lines.append(lines[j])
                if not closing_found:
                    logger.warning(
                        f"</sly-iw-sidebar> closing tag not found after the last h2 header."
                        f" Check that all <h2> headers are presented within <sly-iw-sidebar>"
                        f" to generate sidebar correctly."
                    )
                    new_lines.append("</div>")
        html = "\n".join(new_lines)

        # Generate sidebar
        sidebar_placeholder_pattern = r"^([ \t]*)<!--AUTOSIDEBAR_PLACEHOLDER-->$"
        sidebar_match = re.search(sidebar_placeholder_pattern, html, re.MULTILINE)
        if sidebar_match:
            # base_indent = sidebar_match.group(1)
            base_indent = " " * 4 * 3
            sidebar_parts = []
            for section in headers:
                button_html = (
                    f"{base_indent}<div>\n"
                    f"{base_indent}    <el-button type=\"text\" @click=\"data.scrollIntoView='{section['id']}'\"\n"
                    f"{base_indent}        :style=\"{{fontWeight: data.scrollIntoView === '{section['id']}' ? 'bold' : 'normal'}}\">\n"
                    f"{base_indent}        {section['title']}\n"
                    f"{base_indent}    </el-button>\n"
                    f"{base_indent}</div>"
                )
                sidebar_parts.append(button_html)
            sidebar_code = "\n".join(sidebar_parts)
            html = re.sub(sidebar_placeholder_pattern, sidebar_code, html, flags=re.MULTILINE)

        return html

    def render_to_file(
        self,
        template_path: str,
        context: dict,
        output_path: str,
    ) -> None:
        """
        Renders a template and saves the result to a file.

        :param template_path: Path to the markdown template file relative to template dir.
        :type template_path: str
        :param context: Dictionary with data for template rendering.
        :type context: Dict[str, Any]
        :param output_path: Path to the file where the result will be saved.
        :type output_path: str

        :returns: None
        """
        rendered_content = self.render(template_path, context)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(rendered_content)

    def add_filter(self, filter_name: str, filter_function: callable) -> None:
        """
        Adds a custom filter to the Jinja2 environment.

        :param filter_name: Filter name.
        :type filter_name: str
        :param filter_function: Function that implements the filter.
        :type filter_function: callable

        :returns: None
        """
        self.environment.filters[filter_name] = filter_function

    def add_global(self, variable_name: str, variable_value: Any) -> None:
        """
        Adds a global variable to the Jinja2 environment.

        :param variable_name: Variable name.
        :type variable_name: str
        :param variable_value: Variable value.
        :type variable_value: Any

        :returns: None
        """
        self.environment.globals[variable_name] = variable_value

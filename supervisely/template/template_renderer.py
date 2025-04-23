import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import re
from jinja2 import Environment, FileSystemLoader
from supervisely.template.extensions import MarkdownExtension


class TemplateRenderer:

    def __init__(
        self, 
        md_template_path: str,
        jinja_options: Optional[Dict[str, Any]] = None,
        jinja_extensions: Optional[list] = None,
    ):
        """
        Initializes template renderer with specified parameters.
        
        :param md_template_path: Path to the markdown template file relative to template directory
        :type md_template_path: str
        :param jinja_extensions: List of Jinja2 extensions to use. By default includes MarkdownExtension.
        :type jinja_extensions: Optional[list]
        :param jinja_options: Additional options for configuring Jinja2 environment.
        :type jinja_options: Optional[Dict[str, Any]]
        """
        self.base_template_dir = Path(__file__).parent
        self.layout_template_name = "template.html.jinja"
        self.md_template_path = md_template_path

        if jinja_options is None:
            jinja_options = {
                "autoescape": False,
                "trim_blocks": True,
                "lstrip_blocks": True,
            }

        if jinja_extensions is None:
            jinja_extensions = [MarkdownExtension]

        jinja_extensions = jinja_extensions
        jinja_options = jinja_options
        
        self.env_options = {}
        self.env_options.update(jinja_options)
        self.env_options.update({"extensions": jinja_extensions})

        self.loader = FileSystemLoader(self.base_template_dir)
        self.environment = Environment(loader=self.loader, **self.env_options)
    
    
    def render_template(
        self, 
        template_path: str, 
        context: Dict[str, Any] = None,
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
        if context is None:
            context = {}
        
        markdown_content = self._render_markdown_template(template_path, context)
        navigation, processed_content = self._prepare_markdown_with_sections(markdown_content)
        
        context['navigation'] = {'sections': navigation}
        context['markdown_content'] = processed_content
        
        layout_template = self.environment.get_template(self.layout_template_name)
        return layout_template.render(**context)
    
    def _render_markdown_template(self, template_path: str, context: Dict[str, Any]) -> str:
        """
        Renders a markdown template to process Jinja variables.
        
        :param template_path: Path to the markdown template file relative to template dir.
        :type template_path: str
        :param context: Context for Jinja template rendering.
        :type context: Dict[str, Any]
        
        :returns: Processed markdown content with variables rendered
        :rtype: str
        """
        md_template = self.environment.get_template(template_path)
        markdown_content = md_template.render(**context)
        return markdown_content
    
    def _prepare_markdown_with_sections(self, html_content: str) -> Tuple[List[Dict[str, str]], str]:
        """
        Prepares HTML content by adding section markers for navigation.
        
        :param html_content: HTML content with processed Jinja variables.
        :type html_content: str
        
        :returns: Tuple containing navigation list and modified HTML content with section markers
        :rtype: Tuple[List[Dict[str, str]], str]
        """
        h2_pattern = r'<h2>(.*?)</h2>'
        headers = []
        navigation = []
        
        lines = html_content.split('\n')
        for i, line in enumerate(lines):
            match = re.search(h2_pattern, line)
            if match:
                title = match.group(1).strip()
                section_id = f"{title.lower().replace(' ', '-')}-markdown-section"
                headers.append({
                    'index': i,
                    'title': title,
                    'id': section_id
                })
                navigation.append({
                    'id': section_id,
                    'title': title
                })
        
        if not headers:
            return navigation, html_content
              
        new_lines = []
        
        first_header_index = headers[0]['index']
        new_lines.extend(lines[:first_header_index])
        
        for i, header in enumerate(headers):
            new_lines.append(f'<div id="{header["id"]}" class="section">')
            new_lines.append(lines[header['index']])
            
            if i < len(headers) - 1:
                next_header_index = headers[i + 1]['index']
                new_lines.extend(lines[header['index'] + 1:next_header_index])
                new_lines.append('</div>')
            else:
                new_lines.extend(lines[header['index'] + 1:])
                new_lines.append('</div>')
        
        processed_html = '\n'.join(new_lines)
        return navigation, processed_html
    
    def render_to_file(
        self, 
        template_path: str, 
        output_file_path: str, 
        context: Dict[str, Any] = None,
    ) -> None:
        """
        Renders a template and saves the result to a file.
        
        :param template_path: Path to the markdown template file relative to template dir.
        :type template_path: str
        :param output_file_path: Path to the file where the result will be saved.
        :type output_file_path: str
        :param context: Dictionary with data for template rendering.
        :type context: Dict[str, Any]
        
        :returns: None
        """
        if context is None:
            context = {}
            
        rendered_content = self.render_template(template_path, context)
        output_path = Path(output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
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
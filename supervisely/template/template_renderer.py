from pathlib import Path
from typing import Dict, Any, Optional, Union
from jinja2 import Environment, FileSystemLoader
from supervisely.template.extensions import MarkdownExtension


class TemplateRenderer:

    def __init__(
        self, 
        templates_dir: Optional[Union[str, Path]] = None,
        jinja_options: Optional[Dict[str, Any]] = None,
        jinja_extensions: Optional[list] = None,
    ):
        """
        Initializes template renderer with specified parameters.
        
        :param templates_dir: Path to the templates directory. If None, templates can be loaded directly as strings.
        :type templates_dir: Optional[Union[str, Path]]
        :param jinja_extensions: List of Jinja2 extensions to use. By default includes MarkdownExtension.
        :type jinja_extensions: Optional[list]
        :param jinja_options: Additional options for configuring Jinja2 environment.
        :type jinja_options: Optional[Dict[str, Any]]
        """
        self.templates_dir = templates_dir

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

        self.loader = FileSystemLoader(templates_dir)
        self.environment = Environment(loader=self.loader, **self.env_options)
    
    
    def render_template(
        self, 
        template_path: str, 
        context: Dict[str, Any] = None,
    ) -> str:
        """
        Renders a template with the provided context.
        
        :param template_path: Path to the template file relative to templates_dir.
        :type template_path: str
        :param context: Dictionary with data for template rendering.
        :type context: Dict[str, Any]
        
        :returns: Result of template rendering as a string.
        :rtype: str
        """
        if context is None:
            context = {}
        
        template = self.environment.get_template(template_path)
        return template.render(**context)
    
    def render_to_file(
        self, 
        template_path: str, 
        output_path: str, 
        context: Dict[str, Any] = None,
    ) -> None:
        """
        Renders a template and saves the result to a file.
        
        :param template_path: Path to the template file relative to templates_dir.
        :type template_path: str
        :param output_path: Path to the file where the result will be saved.
        :type output_path: str
        :param context: Dictionary with data for template rendering.
        :type context: Dict[str, Any]
        
        :returns: None
        """
        rendered_content = self.render_template(template_path, context)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered_content)
            
    def add_filter(self, name: str, function: callable) -> None:
        """
        Adds a custom filter to the Jinja2 environment.
        
        :param name: Filter name.
        :type name: str
        :param function: Function that implements the filter.
        :type function: callable
        
        :returns: None
        """
        self.environment.filters[name] = function
        
    def add_global(self, name: str, value: Any) -> None:
        """
        Adds a global variable to the Jinja2 environment.
        
        :param name: Variable name.
        :type name: str
        :param value: Variable value.
        :type value: Any
        
        :returns: None
        """
        self.environment.globals[name] = value
from supervisely.app.singleton import Singleton


class JinjaWidgets(metaclass=Singleton):
    """Singleton context provider for Jinja2 templates with widget scripts and auto-assigned widget IDs."""

    def __init__(self, auto_widget_id=True):
        self.auto_widget_id = auto_widget_id
        self.context = {}
        self.context["__no_html_mode__"] = auto_widget_id
        self.context["__widget_scripts__"] = {}

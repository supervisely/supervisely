from supervisely.app.singleton import Singleton


class JinjaWidgets(metaclass=Singleton):
    def __init__(self, auto_widget_id=True):
        self.auto_widget_id = auto_widget_id
        self.context = {}
        self.context["__no_html_mode__"] = auto_widget_id
        self.context["__widget_scripts__"] = {}
        self.context["__widget_styles__"] = {}

    def add_widget_style(self, widget_name: str, style: str):
        if widget_name not in self.context["__widget_styles__"]:
            self.context["__widget_styles__"][widget_name] = style

    def get_all_styles(self):
        return self.context["__widget_styles__"]

    @property
    def styles_str(self):
        styles = self.get_all_styles()
        if not styles:
            return ""
        return "\n".join(list(styles.values()))

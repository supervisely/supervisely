from supervisely.app.singleton import Singleton


class JinjaWidgets(metaclass=Singleton):
    def __init__(self, auto_widget_id=True):
        self.auto_widget_id = auto_widget_id
        self.context = {}
        self.context["__no_html_mode__"] = auto_widget_id
        self.context["__widget_scripts__"] = {}
        self.context["__widget_styles__"] = {}

    def add_widget_style(self, widget_name: str, styles: list):
        if widget_name not in self.context["__widget_styles__"]:
            self.context["__widget_styles__"][widget_name] = styles
        else:
            for link in styles:
                if link not in self.context["__widget_styles__"][widget_name]:
                    self.context["__widget_styles__"][widget_name].append(link)

    def get_all_styles(self):
        return [l for s in self.context["__widget_styles__"].values() for l in s]

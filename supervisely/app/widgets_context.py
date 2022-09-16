from supervisely.app.singleton import Singleton


class JinjaWidgets(metaclass=Singleton):
    def __init__(self, auto_widget_id=True):
        self.auto_widget_id = auto_widget_id
        self.context = {}

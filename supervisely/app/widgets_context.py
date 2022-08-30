from supervisely.app.singleton import Singleton


class JinjaWidgets(metaclass=Singleton):
    def __init__(self):
        self.context = {}

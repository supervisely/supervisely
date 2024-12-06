from typing import Optional

from supervisely._utils import camel_to_snake, rand_str


class BaseWidget:
    def __init__(self, name: str = None, title: Optional[str] = None) -> None:
        self.type = camel_to_snake(self.__class__.__name__)
        self.id = f"{self.type}_{rand_str(5)}"
        self.name = name
        self.title = title
        self.data_source = f"/data/{self.name}_{self.id}.json"
        self.click_data_source = f"/data/{self.name}_{self.id}_click_data.json"
        self.switchable = False

    def save_data(self, basepath: str) -> None:
        raise NotImplementedError

    def get_state(self) -> None:
        raise NotImplementedError

    def to_html(self) -> str:
        raise NotImplementedError

    def set_click_data(self, click_gallery_id, click_data) -> None:
        return

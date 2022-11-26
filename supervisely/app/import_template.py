try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal


class Import:
    def __init__(
        self,
    ):
        pass

    def process_file(path: str):
        pass

    def process_folder(path: str):
        pass

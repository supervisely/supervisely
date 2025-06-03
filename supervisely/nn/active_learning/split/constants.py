from enum import Enum, auto
from typing import List, TypedDict, Union


class SplitMode(Enum):
    COLLECTIONS = "collections"
    DATASETS = "datasets"
    RANDOM = "random"


class SplitSettings(TypedDict):
    mode: str
    train: Union[float, List[str]]
    val: Union[float, List[str]]

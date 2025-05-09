from enum import Enum, auto
from typing import List, NotRequired, TypedDict


class SamplingSettings(TypedDict, total=False):
    mode: str
    sample_size: NotRequired[int]
    diversity_mode: NotRequired[str]
    prompt: NotRequired[str]


class SamplingMode(Enum):
    RANDOM = "Random"
    DIVERSE = "Diverse"
    AI_SEARCH = "AI Search"

# coding: utf-8

from enum import Enum


class StrEnum(Enum):
    def __str__(self):
        return str(self.value)

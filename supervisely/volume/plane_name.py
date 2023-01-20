# coding: utf-8

from supervisely.collection.str_enum import StrEnum


class PlaneName(StrEnum):
    SAGITTAL = "x"
    CORONAL = "y"
    AXIAL = "z"

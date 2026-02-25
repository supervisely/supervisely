# coding: utf-8
from supervisely.collection.str_enum import StrEnum


# ['admin',  'developer', 'manager', 'reviewer', 'annotator', 'viewer']
class UserRoleName(StrEnum):
    """
    Enumerates supported Supervisely user roles (e.g. admin, manager, annotator).

    You can learn more about the roles in the `User Roles documentation <https://docs.supervisely.com/collaboration/members>`_.
    """
    ADMIN = 'admin'
    """Has full access in the team. Admin can invite new team members and remove entities created by the other team members."""
    DEVELOPER = 'developer'
    """Similar to the admin, but can only remove entities created by themself and cannot invite new members to the team."""
    MANAGER = 'manager'
    """Has no access to things like Neural Networks, but can view and modify Projects & Labeling Jobs."""
    REVIEWER = 'reviewer'
    """Same as Annotator, but can also create new labeling jobs."""
    ANNOTATOR = 'annotator'
    """Has access only to a single page, Labeling Jobs."""
    VIEWER = 'viewer'
    """Can only view items in team."""
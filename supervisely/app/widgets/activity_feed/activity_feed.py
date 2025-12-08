from __future__ import annotations

from typing import Dict, List, Optional

from supervisely.app import DataJson
from supervisely.app.widgets import Widget

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class ActivityFeed(Widget):
    """ActivityFeed is a widget that displays a vertical list of activity items with status indicators.
    Similar to a timeline or activity log showing sequential events with their current status.

    Each item can contain a custom widget as content and displays a status indicator (pending, in process, completed, failed).
    Items are automatically numbered if no number is provided.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/layouts-and-containers/activity-feed>`_
        (including screenshots and examples).

    :param items: List of ActivityFeed.Item objects to display
    :type items: Optional[List[ActivityFeed.Item]]
    :param widget_id: An identifier of the widget.
    :type widget_id: str, optional

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import ActivityFeed, Text

        # Create items with custom content
        item1 = ActivityFeed.Item(
            content=Text("Processing dataset"),
            status="completed"
        )
        item2 = ActivityFeed.Item(
            content=Text("Training model"),
            status="in_progress",
            number=2
        )
        item3 = ActivityFeed.Item(
            content=Text("Generating report"),
            status="pending"
        )

        # Create activity feed
        feed = ActivityFeed(items=[item1, item2, item3])

        # Add item during runtime
        new_item = ActivityFeed.Item(
            content=Text("Deploy model"),
            status="pending"
        )
        feed.add_item(new_item)

        # Update status by item number
        feed.set_status(2, "completed")

        # Get item status
        status = feed.get_status(2)
    """

    class Item:
        """Represents a single item in the ActivityFeed.

        :param content: Widget to display as the item content
        :type content: Widget
        :param status: Status of the item (pending, in_progress, completed, failed)
        :type status: Literal["pending", "in_progress", "completed", "failed"]
        :param number: Position number in the feed (auto-assigned if not provided)
        :type number: Optional[int]
        """

        def __init__(
            self,
            content: Widget,
            status: Literal["pending", "in_progress", "completed", "failed"] = "pending",
            number: Optional[int] = None,
        ) -> ActivityFeed.Item:
            self.content = content
            self.status = status
            self.number = number
            self._validate_status()

        def _validate_status(self):
            valid_statuses = ["pending", "in_progress", "completed", "failed"]
            if self.status not in valid_statuses:
                raise ValueError(
                    f"Invalid status '{self.status}'. Must be one of: {', '.join(valid_statuses)}"
                )

        def to_json(self):
            return {
                "number": self.number,
                "status": self.status,
            }

    def __init__(
        self,
        items: Optional[List[ActivityFeed.Item]] = None,
        widget_id: Optional[str] = None,
    ):
        self._items = items if items is not None else []
        self._auto_assign_numbers()
        super().__init__(widget_id=widget_id, file_path=__file__)

    def _auto_assign_numbers(self):
        """Automatically assign numbers to items that don't have them."""
        next_number = 1
        for item in self._items:
            if item.number is None:
                item.number = next_number
            next_number = max(next_number, item.number) + 1

    def get_json_data(self) -> Dict:
        """Returns dictionary with widget data.

        :return: Dictionary with items data
        :rtype: Dict
        """
        return {
            "items": [item.to_json() for item in self._items],
        }

    def get_json_state(self) -> Dict:
        """Returns dictionary with widget state (empty for this widget).

        :return: Empty dictionary
        :rtype: Dict
        """
        return {}

    def add_item(
        self,
        item: Optional[ActivityFeed.Item] = None,
        content: Optional[Widget] = None,
        status: Literal["pending", "in_progress", "completed", "failed"] = "pending",
        number: Optional[int] = None,
    ) -> None:
        """Add a new item to the activity feed.

        You can either pass an ActivityFeed.Item object or provide content and status separately.

        :param item: ActivityFeed.Item to add
        :type item: Optional[ActivityFeed.Item]
        :param content: Widget content (used if item is not provided)
        :type content: Optional[Widget]
        :param status: Status of the item (used if item is not provided)
        :type status: Literal["pending", "in_progress", "completed", "failed"]
        :param number: Position number (auto-assigned if not provided)
        :type number: Optional[int]
        """
        if item is None:
            if content is None:
                raise ValueError("Either 'item' or 'content' must be provided")
            item = ActivityFeed.Item(content=content, status=status, number=number)

        if item.number is None:
            # Auto-assign number
            if self._items:
                item.number = max(i.number for i in self._items) + 1
            else:
                item.number = 1

        self._items.append(item)
        self.update_data()
        DataJson().send_changes()

    def remove_item(self, number: int) -> None:
        """Remove an item from the activity feed by its number.

        :param number: Number of the item to remove. Starts from 1.
        :type number: int
        """
        self._items = [item for item in self._items if item.number != number]
        self.update_data()
        DataJson().send_changes()

    def set_status(
        self,
        number: int,
        status: Literal["pending", "in_progress", "completed", "failed"],
    ) -> None:
        """Update the status of an item by its number.

        :param number: Number of the item to update. Starts from 1.
        :type number: int
        :param status: New status for the item
        :type status: Literal["pending", "in_progress", "completed", "failed"]
        """
        for i, item in enumerate(self._items):
            if item.number == number:
                item.status = status
                item._validate_status()
                DataJson()[self.widget_id]["items"][i]["status"] = status
                DataJson().send_changes()
                return
        raise ValueError(f"Item with number {number} not found")

    def get_status(self, number: int) -> str:
        """Get the status of an item by its number.

        :param number: Number of the item. Starts from 1.
        :type number: int
        :return: Status of the item
        :rtype: str
        """
        for item in self._items:
            if item.number == number:
                return item.status
        raise ValueError(f"Item with number {number} not found")

    def get_items(self) -> List[ActivityFeed.Item]:
        """Get all items in the activity feed.

        :return: List of all items
        :rtype: List[ActivityFeed.Item]
        """
        return self._items

    def clear(self) -> None:
        """Remove all items from the activity feed."""
        self._items = []
        self.update_data()
        DataJson().send_changes()

    def set_items(self, items: List[ActivityFeed.Item]) -> None:
        """Replace all items in the activity feed.

        :param items: New list of items
        :type items: List[ActivityFeed.Item]
        """
        self._items = items
        self._auto_assign_numbers()
        self.update_data()
        DataJson().send_changes()

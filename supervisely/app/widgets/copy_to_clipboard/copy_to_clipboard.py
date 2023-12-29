from typing import Dict, Optional, Union

from supervisely.app.widgets import Editor, Input, Text, TextArea, Widget


class CopyToClipboard(Widget):
    """CopyToClipboard widget allows you to wrap your widgets (Editor, Text, TextArea, or Input) and str text with a copy button.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/text-elements/copytoclipboard>`_
        (including screenshots and examples).

    :param content: content to be copied
    :type content: Union[Editor, Text, TextArea, Input, str]
    :param widget_id: An identifier of the widget.
    :type widget_id: str, optional
    :raises TypeError: if content is not str, Editor, Text, TextArea, or Input

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import CopyToClipboard, Text

        text = Text("Text to be copied")
        copy_to_clipboard = CopyToClipboard(text)
    """

    def __init__(
        self,
        content: Optional[Union[Editor, Text, TextArea, Input, str]] = "",
        widget_id: Optional[str] = None,
    ):
        self._content = content

        if not isinstance(content, (str, Editor, Text, TextArea, Input)):
            raise TypeError(
                f"Supported types: str, Editor, Text, TextArea, Input. Your type: {type(content).__name__}"
            )
        if isinstance(content, str):
            self._content_widget_type = "str"
            self._curr_prop_name = None
            self._content_value = content
        else:
            if isinstance(content, (Editor, Input)):
                self._content_widget_type = "input"
                self._curr_prop_name = "value" if isinstance(content, Input) else "text"
            elif isinstance(content, (Text, TextArea)):
                self._content_widget_type = "text"
                self._curr_prop_name = "value" if isinstance(content, TextArea) else "text"
            self._content_value = content.get_value()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict[str, Union[str, Dict]]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - content: content to be copied
            - curr_property: current property of the content widget

        :return: dictionary with widget data
        :rtype: Dict[str, Union[str, Dict]]
        """
        return {"content": self._content_value, "curr_property": self._curr_prop_name}

    def get_json_state(self) -> Dict[str, Union[str, Dict]]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
            - content: content to be copied
            - curr_property: current property of the content widget

        :return: dictionary with widget state
        :rtype: Dict[str, Union[str, Dict]]
        """
        return {"content": self._content_value, "curr_property": self._curr_prop_name}

    def get_content(self) -> Union[Editor, Input, Text, TextArea, str]:
        """Returns content of the widget.

        :return: content of the widget
        :rtype: Union[Editor, Input, Text, TextArea, str]
        """
        return self._content

from typing import Literal
from pathlib import Path
from jinja2 import Environment
import markupsafe
from supervisely.app.jinja2 import create_env
from supervisely.app import DataJson
from supervisely.app.fastapi import Jinja2Templates

INFO = "info"
WARNING = "warning"
ERROR = "error"


class NotificationBox:
    def __init__(
        self,
        widget_id: str,
        title: str = None,
        description: str = None,
        box_type: Literal["info", "warning", "error"] = WARNING,
    ):
        self.widget_id = widget_id
        self.auto_registration = True
        self.title = title
        self.description = description
        if self.title is None and self.description is None:
            raise ValueError(
                "Both title and description can not be None at the same time"
            )
        self.box_type = box_type
        self.icon = "zmdi-alert-triangle"  # @TODO: get by box type
        if self.box_type != WARNING:
            raise ValueError(
                f"Only {WARNING} type is supported. Other types {[INFO, WARNING, ERROR]} will be supported later"
            )

        if self.auto_registration is True:
            self.init(DataJson(), Jinja2Templates())

    def init(self, data: DataJson, templates: Jinja2Templates):
        data.raise_for_key(self.widget_id)
        data[self.widget_id] = {
            "title": self.title,
            "description": self.description,
            "icon": self.icon,
        }
        templates.context_widgets[self.widget_id] = self

    def to_html(self):
        current_dir = Path(__file__).parent.absolute()
        jinja2_sly_env: Environment = create_env(current_dir)
        html = jinja2_sly_env.get_template("notification_box.html").render(
            {"widget": self}
        )
        return markupsafe.Markup(html)


# <div>
#     <div
#       v-if="data.{{{widget.widget_id}}}.title"
#       class="notification-box-title"
#       v-html="{{{widget.title}}}"
#     ></div>
#     <div
#       v-if="data.{{{widget.widget_id}}}.description"
#       v-html="{{{widget.description}}}"
#     ></div>
#   </div>


# <div class="notification-box notification-box-{{{widget.box_type}}}">
#   <i class="notification-box-icon zmdi {{{widget.icon}}}"></i>

#   <div>
#     <div
#       v-if="data.{{{widget.widget_id}}}.title"
#       class="notification-box-title"
#       v-html="{{{widget.title}}}"
#     ></div>
#     <div
#       v-if="data.{{{widget.widget_id}}}.description"
#       v-html="{{{widget.description}}}"
#     ></div>
#   </div>
# </div>

# <div v-if="data.{{{widget.widget_id}}}.description">
#       {{{widget.description}}}
#     </div>

# <sly-style>
#   <!-- <style> -->
#   .notification-box { background-color: lightgray; padding: 10px; border-radius:
#   6px; display: flex; align-items: center; } .notification-box-warning {
#   background-color: #fcf8e3; border: 1px solid #faebcc; border-left: 4px solid
#   #cd9c38; } .notification-box-icon { font-size: 20px; margin-right: 10px; }
#   .notification-box-warning .notification-box-icon { color: #cd9c38; }
#   .notification-box-title { font-size: 16px; font-weight: bold; }
#   <!-- </style> -->
# </sly-style>

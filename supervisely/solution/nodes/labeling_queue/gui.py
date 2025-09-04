from supervisely._utils import abs_url
from supervisely.app.widgets import (
    Button,
    Container,
    Dialog,
    Field,
    MembersListSelector,
)


class LabelingQueueGUI:
    def __init__(self, queue_id: int):
        self.queue_id = queue_id

        # self.add_annotator_btn = Button(
        #     "Add annotator",
        #     icon="zmdi zmdi-accounts-add",
        #     button_size="mini",
        #     plain=True,
        #     button_type="text",
        # )
        # self.add_reviewer_btn = Button(
        #     "Add reviewer",
        #     icon="zmdi zmdi-account-add",
        #     button_size="mini",
        #     plain=True,
        #     button_type="text",
        # )

        # self.user_selector = MembersListSelector(multiple=True)
        # self.annotators = Field(
        #     title="Select annotators to add",
        #     content=self.user_selector,
        # )
        # self.annotators.hide()
        # self.reviewers = Field(title="Select reviewers to add", content=self.user_selector)
        # self.reviewers.hide()
        # self.add_user_modal = Dialog(
        #     title="Add user",
        #     content=Container([self.annotators, self.reviewers], gap=0),
        # )

    @property
    def open_labeling_queue_btn(self):
        if not hasattr(self, "_open_labeling_queue_btn"):
            link = abs_url(f"labeling/jobs/list?queueId={self.queue_id}")
            self._open_labeling_queue_btn = Button(
                "Open Labeling Queue",
                icon="mdi mdi-open-in-new",
                button_size="mini",
                link=link,
                plain=True,
                button_type="text",
            )
        return self._open_labeling_queue_btn

    @property
    def debug_btn_1(self):
        if not hasattr(self, "_debug_btn_1"):
            self._debug_btn_1 = Button(
                "Predict & Accept all",
                icon="mdi mdi-bug-play",
                button_size="mini",
                plain=True,
                button_type="text",
            )
        return self._debug_btn_1

    @property
    def debug_btn_2(self):
        if not hasattr(self, "_debug_btn_2"):
            self._debug_btn_2 = Button(
                "Accept all",
                icon="mdi mdi-bug-play-outline",
                button_size="mini",
                plain=True,
                button_type="text",
            )
        return self._debug_btn_2

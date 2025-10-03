from supervisely.app.widgets import Button, TasksHistory
from supervisely.app.widgets.dialog.dialog import Dialog
from supervisely.app.widgets.fast_table.fast_table import FastTable
from supervisely.solution.engine.modal_registry import ModalRegistry


class TasksHistoryWidget(TasksHistory):
    def __init__(self, *args, **kwargs):
        self._modal = None
        self._logs_modal = None
        self._preview_modal = None
        super().__init__(*args, **kwargs)

        # --- modal ----------------------------------------------------
        ModalRegistry().attach_history_widget(owner_id=self.widget_id, widget=self.table)
        ModalRegistry().attach_preview_widget(owner_id=self.widget_id, widget=self.gallery)
        ModalRegistry().attach_logs_widget(owner_id=self.widget_id, widget=self.logs)

    def _on_table_row_click(self, clicked_row: FastTable.ClickedRow):
        # can be re-implemented in child classes (for example, to show preview of selected task)
        self.logs.set_task_id(clicked_row.row[0])
        ModalRegistry().open_logs(owner_id=self.widget_id)

    @property
    def logs_modal(self) -> Dialog:
        if self._logs_modal is None:
            self._logs_modal = ModalRegistry().logs_dialog
        return self._logs_modal

    @property
    def preview_modal(self) -> Dialog:
        if self._preview_modal is None:
            self._preview_modal = ModalRegistry().preview_dialog
        return self._preview_modal

    @property
    def modal(self) -> Dialog:
        if self._modal is None:
            self._modal = ModalRegistry().history_dialog
        return self._modal

    @property
    def open_modal_button(self) -> Button:
        """Small button that opens the history modal."""
        if not hasattr(self, "_open_modal_button") or self._open_modal_button is None:
            btn = Button(
                text="Tasks History",
                icon="zmdi zmdi-format-list-bulleted",
                button_size="mini",
                plain=True,
                button_type="text",
            )

            @btn.click
            def _on_click():
                self.update()
                ModalRegistry().open_history(owner_id=self.widget_id)

            self._open_modal_button = btn
        return self._open_modal_button

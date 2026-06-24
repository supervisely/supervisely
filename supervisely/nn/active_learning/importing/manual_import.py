from typing import TYPE_CHECKING, Optional

from supervisely.nn.active_learning.session import ActiveLearningSession

from supervisely.nn.active_learning.utils.constants import AUTOIMPORT_SLUG

def filter_tasks_by_slug(import_history: list, app_slug: str):
    tasks = []
    for history_item in import_history:
        if history_item.get("slug") == app_slug:
            tasks.append(history_item.get("task_id"))
    return tasks


def get_last_imported_images_count(
    al_session: ActiveLearningSession, slug: Optional[str] = None
) -> Optional[int]:
    """
    Get the number of images imported in the last import task.
    This function checks the import history of the input project and returns the count of images
    imported in the last task with the specified slug.
    """
    al_session.state.refresh()
    input_project = al_session.state.project
    import_history = input_project.custom_data.get("import_history", {}).get("tasks", [])

    solutions_autoimport_tasks = al_session.state.import_tasks.get(slug, [])

    # update auto import tasks button
    auto_import_tasks = filter_tasks_by_slug(import_history, slug)
    if any([task not in solutions_autoimport_tasks for task in auto_import_tasks]):
        al_session.state.add_import_tasks(slug, auto_import_tasks)

    # get last imported images count
    last_imported_images_count = None
    for import_task in import_history[::-1]:
        if len(solutions_autoimport_tasks) > 0:
            if import_task.get("task_id") == solutions_autoimport_tasks[-1]:
                last_imported_images_count = import_task.get("items_count", 0)
                break
    return last_imported_images_count

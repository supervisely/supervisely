import sys

_import_failed = False
try:
    from supervisely.app.content import StateJson, DataJson
    from supervisely.app.content import get_data_dir
    import supervisely.app.fastapi as fastapi
    import supervisely.app.widgets as widgets
except (ImportError, ModuleNotFoundError) as e:
    _import_failed = True
    pass


def __getattr__(name):
    if _import_failed is True:
        raise ModuleNotFoundError(
            'No module named supervisely.app, please install dependencies with "pip install supervisely[apps]"'
        )
    return getattr(sys.modules[__name__], name)

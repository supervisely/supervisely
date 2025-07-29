from supervisely.sly_logger import logger

try:
    from supervisely.nn.tracker.bot_sort import BoTTracker
    from supervisely.nn.tracker.deep_sort import DeepSortTracker
except ImportError:
    logger.error(
        "Failed to import tracker modules. Please try install extras with 'pip install supervisely[tracking]'"
    )
    raise

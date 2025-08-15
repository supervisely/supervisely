from supervisely.sly_logger import logger

try:
    from supervisely.nn.tracker.botsort_tracker import BotSortTracker
except ImportError:
    logger.error(
        "Failed to import tracker modules. Please try install extras with 'pip install supervisely[tracking]'"
    )
    raise

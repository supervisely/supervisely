from supervisely.sly_logger import logger

try:
    from supervisely.nn.tracker.botsort_tracker import BotSortTracker
    from supervisely.nn.tracker.calculate_metrics import TrackingEvaluator, evaluate
    from supervisely.nn.tracker.visualize import TrackingVisualizer, visualize
except ImportError:
    logger.error(
        "Failed to import tracker modules. Please try install extras with 'pip install supervisely[tracking]'"
    )
    raise

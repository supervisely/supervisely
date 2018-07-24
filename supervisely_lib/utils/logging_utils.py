# coding: utf-8

import sys
import os
from ..sly_logger import logger, EventType


def main_wrapper(main_name, main_func, *args, **kwargs):
    try:
        logger.debug('Main started.', extra={'main_name': main_name})
        main_func(*args, **kwargs)
    except Exception:
        logger.fatal('Unexpected exception in main.', extra={'main_name': main_name, 'event_type': EventType.TASK_CRASHED}, exc_info=True)
        logger.debug('Main finished: BAD.', extra={'main_name': main_name})
        #sys.exit(70)
        os._exit(1)
    else:
        logger.debug('Main finished: OK.', extra={'main_name': main_name})

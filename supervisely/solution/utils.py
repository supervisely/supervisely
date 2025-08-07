from typing import Tuple


def get_interval_period(sec: int) -> Tuple[str, int]:
    """
    Returns the period and interval based on the given seconds.
    :param sec: Number of seconds
    :return: Tuple of (period, interval)
    """
    if sec is None:
        return None, None
    if sec < 60:
        period = "s"
        interval = sec
    elif sec // 60 < 60:
        period = "min"
        interval = sec // 60
    elif sec // 3600 < 24:
        period = "h"
        interval = sec // 3600
    else:
        period = "d"
        interval = sec // 86400
    return period, interval


def get_seconds_from_period_and_interval(period: str, interval: int) -> int:
    """
    Returns the number of seconds based on the given period and interval.
    :param period: Period of time ('min', 'h', 'd')
    :param interval: Interval value
    :return: Number of seconds
    """
    if period == "min":
        return interval * 60
    elif period == "h":
        return interval * 3600
    elif period == "d":
        return interval * 86400
    else:
        raise ValueError(f"Unknown period: {period}")

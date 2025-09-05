from typing import Tuple

from supervisely.api.api import Api


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

def find_agent(api: Api, team_id: int) -> int:
    agents = api.agent.get_list_available(team_id, show_public=True, has_gpu=True)
    if len(agents) == 0:
        raise ValueError("No available agents found.")
    agent_id_memory_map = {}
    kubernetes_agents = []
    for agent in agents:
        if agent.type == "sly_agent":
            # No multi-gpu support, always take the first one
            agent_id_memory_map[agent.id] = agent.gpu_info["device_memory"][0]["available"]
        elif agent.type == "kubernetes":
            kubernetes_agents.append(agent.id)
    if len(agent_id_memory_map) > 0:
        return max(agent_id_memory_map, key=agent_id_memory_map.get)
    if len(kubernetes_agents) > 0:
        return kubernetes_agents[0]

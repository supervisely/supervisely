import os


def _int_from_env(value):
    if value is None:
        return value
    return int(value)


def agent_id():
    return _int_from_env(os.environ.get("AGENT_ID"))


def agent_storage():
    return os.environ.get("AGENT_STORAGE")

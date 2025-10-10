import supervisely.io.env as sly_env


class Singleton(type):
    _instances = {}
    _nested_instances = {}

    def __call__(cls, *args, **kwargs):
        local = kwargs.pop("__local__", False)
        # user_id_kwarg = kwargs.pop("user_id", None)
        if local is False:
            if cls not in cls._instances:
                cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

            if sly_env.is_multiuser_mode_enabled():
                from copy import deepcopy

                from supervisely.app.content import DataJson, StateJson

                if cls not in cls._nested_instances:
                    cls._nested_instances[cls] = {}
                user_id = sly_env.user_from_multiuser_app()
                # user_id = (
                #     user_id_kwarg
                #     if user_id_kwarg is not None
                #     else sly_env.user_from_multiuser_app()
                # )
                if user_id is not None and (cls in (StateJson, DataJson)):
                    if user_id not in cls._nested_instances[cls]:
                        cls._nested_instances[cls][user_id] = super(Singleton, cls).__call__(
                            *args, **kwargs
                        )
                        cls._nested_instances[cls][user_id].update(dict(cls._instances[cls]))

                    return cls._nested_instances[cls][user_id]
            return cls._instances[cls]
        else:
            return super(Singleton, cls).__call__(*args, **kwargs)

import supervisely.io.env as sly_env


class Singleton(type):
    _instances = {}
    _nested_instances = {}

    def __call__(cls, *args, **kwargs):
        local = kwargs.pop("__local__", False)
        if local is False:
            if cls not in cls._instances:
                cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

            if sly_env.is_multiuser_mode_enabled():
                from supervisely.app.content import DataJson, StateJson
                from copy import deepcopy

                # Initialize nested instances dict once
                nested_instances = cls._nested_instances.setdefault(cls, {})
                
                user_id = sly_env.user_from_multiuser_app()
                if user_id is not None and cls in (StateJson, DataJson):
                    if user_id not in nested_instances:
                        # Create new instance and copy data
                        instance = super(Singleton, cls).__call__(*args, **kwargs)
                        instance.update(deepcopy(dict(cls._instances[cls])))
                        nested_instances[user_id] = instance
                    
                    return nested_instances[user_id]
            return cls._instances[cls]
        else:
            return super(Singleton, cls).__call__(*args, **kwargs)

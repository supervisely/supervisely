# coding: utf-8
"""Backward compatibility for objects pickled by older SDK versions.

Pickle restores __dict__ directly and never calls __init__, so an object
pickled before a new attribute existed comes back missing it entirely
(e.g. Data Version restore unpickling an old project backup). Declare such
attributes with @legacy_pickle_defaults to backfill them on unpickle:

    @legacy_pickle_defaults(_target_type=TagTargetType.ALL)
    class TagMeta(KeyObject, JsonSerializable):
        ...

Values may be plain defaults, or callable(self) when computed from other
attributes. When adding an attribute to a class that ends up inside a
pickled payload, declare its default here too.

Backfill only - never re-validate business rules on unpickle. Validation
rules get stricter over time, so re-running today's validation against
yesterday's data would turn a harmless restore into a crash that could not
have happened when that data was created. A deserialized object represents
state that was already valid back then; treat it as historical fact, not
something to re-approve under current rules.
"""

from supervisely.sly_logger import logger

_DEFAULTS_ATTR = "_pickle_defaults"

# Warn once per (class, attribute), not once per unpickled object.
_warned = set()


def _collect_defaults(cls):
    """Merge _pickle_defaults along the MRO so subclasses keep inherited entries."""
    merged = {}
    for klass in reversed(cls.__mro__):
        merged.update(vars(klass).get(_DEFAULTS_ATTR, {}))
    return merged


def _setstate(self, state):
    self.__dict__.update(state)
    for attr, default in _collect_defaults(type(self)).items():
        if attr not in self.__dict__:
            self.__dict__[attr] = default(self) if callable(default) else default
            key = (type(self).__name__, attr)
            if key not in _warned:
                _warned.add(key)
                logger.warning(
                    f"Unpickled a legacy '{key[0]}' object with no '{key[1]}' "
                    "(created by an older SDK version) - backfilled with a default."
                )


def legacy_pickle_defaults(**defaults):
    """Backfill the given attributes when unpickling legacy objects of this class."""

    def decorator(cls):
        if "__setstate__" in vars(cls):
            raise TypeError(
                f"{cls.__name__} defines its own __setstate__; fold its logic into the "
                "@legacy_pickle_defaults declaration instead of using both."
            )
        setattr(cls, _DEFAULTS_ATTR, defaults)
        cls.__setstate__ = _setstate
        return cls

    return decorator

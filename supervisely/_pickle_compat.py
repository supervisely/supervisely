# coding: utf-8
"""Mixin for restoring objects pickled by older SDK versions.

Pickle restores __dict__ directly and never calls __init__, so an object
pickled before a new attribute existed comes back missing it entirely
(e.g. Data Version restore unpickling an old project backup). Classes
opt in by inheriting LegacyPickleCompat and declaring _PICKLE_DEFAULTS:
attr name -> default value, or attr name -> callable(self) for defaults
computed from other attributes.

Backfill only - never re-validate business rules here. Validation rules
can get stricter over time (e.g. a constraint added long after old data
was created), so re-running today's validation against yesterday's data
would turn a harmless restore into a crash that couldn't have happened
before. A deserialized object represents state that was already valid
when it was created; treat it as historical fact, not something to
re-approve under current rules.
"""

from supervisely.sly_logger import logger


class LegacyPickleCompat:
    """Backfills attributes missing from legacy pickles. See module docstring."""

    _PICKLE_DEFAULTS = {}
    _warned = set()

    def __setstate__(self, state):
        self.__dict__.update(state)
        for attr, default in self._collect_pickle_defaults().items():
            if attr not in self.__dict__:
                self.__dict__[attr] = default(self) if callable(default) else default
                key = (type(self).__name__, attr)
                if key not in LegacyPickleCompat._warned:
                    LegacyPickleCompat._warned.add(key)
                    logger.warning(
                        f"Unpickled a legacy '{key[0]}' object missing '{key[1]}' "
                        "(created by an older SDK version) - backfilled with a default."
                    )

    @classmethod
    def _collect_pickle_defaults(cls):
        merged = {}
        for klass in reversed(cls.__mro__):
            merged.update(vars(klass).get("_PICKLE_DEFAULTS", {}))
        return merged

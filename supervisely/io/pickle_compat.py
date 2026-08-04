# coding: utf-8
"""Backward compatibility for objects restored from pickles written by older SDK versions.

Pickle restores __dict__ directly and never calls __init__, so an object
pickled before a new attribute existed comes back missing it entirely
(e.g. Data Version restore reading an old project backup). Declare such
attributes with @legacy_pickle_defaults:

    @legacy_pickle_defaults(_target_type=TagTargetType.ALL)
    class TagMeta(KeyObject, JsonSerializable):
        ...

Values may be plain defaults, or callable(obj) when computed from other
attributes - so a default that is itself a callable value is not supported.
When adding an attribute to a class that ends up inside a pickled payload,
declare its default here too.

The decorator only records the declaration; it installs no __setstate__ and
leaves the class otherwise untouched, so nothing changes for normal use,
copying or pickling. The backfill is applied explicitly by whoever loads a
legacy payload, by calling restore_legacy_defaults() on the objects it
reconstructed - see Project.upload_bin(). Only classes whose instances keep
state in __dict__ are supported.

Backfill only - never re-validate business rules on restore. Validation
rules get stricter over time, so re-running today's validation against
yesterday's data would turn a harmless restore into a crash that could not
have happened when that data was created. A deserialized object represents
state that was already valid back then; treat it as historical fact, not
something to re-approve under current rules.
"""

from supervisely.sly_logger import logger

_DEFAULTS_ATTR = "_pickle_defaults"

# Warn once per (class, attribute), not once per restored object.
_warned = set()


def _collect_defaults(cls):
    """Merge _pickle_defaults along the MRO so subclasses keep inherited entries."""
    merged = {}
    for klass in reversed(cls.__mro__):
        merged.update(vars(klass).get(_DEFAULTS_ATTR, {}))
    return merged


def legacy_pickle_defaults(**defaults):
    """Declare defaults used to backfill this class when restoring a legacy pickle."""

    def decorator(cls):
        setattr(cls, _DEFAULTS_ATTR, defaults)
        return cls

    return decorator


def restore_legacy_defaults(*objects):
    """Backfill declared attributes missing from objects restored from an old pickle.

    Pass the objects reconstructed from the payload; objects whose class
    declares no defaults, and attributes already present, are left alone.
    """
    for obj in objects:
        if obj is None:
            continue
        for attr, default in _collect_defaults(type(obj)).items():
            if attr in obj.__dict__:
                continue
            obj.__dict__[attr] = default(obj) if callable(default) else default
            key = (type(obj).__name__, attr)
            if key not in _warned:
                _warned.add(key)
                logger.warning(
                    f"Restored a legacy '{key[0]}' object with no '{key[1]}' "
                    "(created by an older SDK version) - backfilled with a default."
                )

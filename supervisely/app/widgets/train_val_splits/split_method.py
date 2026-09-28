from typing import Iterable, List, Optional

from supervisely.sly_logger import logger


class SplitMethod:
    """Canonical names of the train/val split methods.

    The constants below are the titles of the tabs rendered by the ``TrainValSplits``
    widget and, at the same time, the values returned by
    :meth:`TrainValSplits.get_split_method`. Short names are the keys stored in
    ``app_state.json`` and accepted in app options.

    Always compare a split method against these constants (normalizing user input
    with :meth:`parse`) instead of hardcoding a string literal: the tab title
    ("Based on item tags") and the literal used in the validation code
    ("Based on tags") had silently drifted apart, which disabled the tag-based
    split without any error message.
    """

    RANDOM = "Random"
    TAGS = "Based on item tags"
    DATASETS = "Based on datasets"
    COLLECTIONS = "Based on collections"

    ALL = (RANDOM, TAGS, DATASETS, COLLECTIONS)

    # Legacy titles that are still accepted on input (some app options and UIs use them).
    # Kept as constants on purpose: they must stay recognized, but never be produced.
    TAGS_LEGACY = "Based on tags"
    LEGACY_TITLES = {TAGS_LEGACY: TAGS}

    # Short names used in app_state.json and in app options
    SHORT_NAMES = {
        RANDOM: "random",
        TAGS: "tags",
        DATASETS: "datasets",
        COLLECTIONS: "collections",
    }

    # Every spelling ever written in app options / app states, lowercased
    _ALIASES = {
        "random": RANDOM,
        "tags": TAGS,
        TAGS_LEGACY.lower(): TAGS,
        TAGS.lower(): TAGS,
        "datasets": DATASETS,
        "based on datasets": DATASETS,
        "collections": COLLECTIONS,
        "based on collections": COLLECTIONS,
    }

    @classmethod
    def parse(cls, split_method: str, raise_error: bool = True) -> Optional[str]:
        """Normalize any known spelling of a split method to its canonical title.

        :param split_method: Split method: a tab title, a short name or a legacy alias.
        :type split_method: str
        :param raise_error: If True, raise ValueError on an unknown method, otherwise return None.
        :type raise_error: bool
        :returns: One of the :class:`SplitMethod` constants, or None if unknown and raise_error is False.
        :rtype: Optional[str]
        """
        method = cls._ALIASES.get(str(split_method).strip().lower())
        if method is None and raise_error:
            raise ValueError(
                f"Unknown train/val split method: '{split_method}'. "
                f"Expected one of {list(cls.ALL)} or their short names "
                f"{list(cls.SHORT_NAMES.values())}"
            )
        return method

    @classmethod
    def parse_list(cls, split_methods: Iterable[str]) -> List[str]:
        """Normalize a list of split methods, skipping (with a warning) the unknown ones.

        :param split_methods: Split methods to normalize.
        :type split_methods: Iterable[str]
        :returns: Canonical split method titles, without duplicates, in the original order.
        :rtype: List[str]
        """
        methods = []
        for split_method in split_methods or []:
            method = cls.parse(split_method, raise_error=False)
            if method is None:
                logger.warning(
                    f"Unknown train/val split method: '{split_method}'. Skipping it. "
                    f"Expected one of {list(cls.ALL)}"
                )
            elif method not in methods:
                methods.append(method)
        return methods

    @classmethod
    def matches(cls, split_method: str, expected: str) -> bool:
        """Check that a split method is the expected one, ignoring spelling and aliases.

        :param split_method: Split method to check.
        :type split_method: str
        :param expected: Expected split method, usually one of the :class:`SplitMethod` constants.
        :type expected: bool
        :returns: True if both refer to the same split method.
        :rtype: bool
        """
        return cls.parse(split_method, raise_error=False) == cls.parse(expected)

    @classmethod
    def to_short(cls, split_method: str) -> str:
        """Convert a split method to its short name used in app_state.json.

        :param split_method: Split method: a tab title, a short name or a legacy alias.
        :type split_method: str
        :returns: Short name of the split method.
        :rtype: str
        """
        return cls.SHORT_NAMES[cls.parse(split_method)]

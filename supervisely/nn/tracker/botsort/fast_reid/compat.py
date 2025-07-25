"""
Compatibility module for Python 3.10+ collections imports
"""

try:
    from collections.abc import Mapping, MutableMapping
except ImportError:
    try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
, MutableMapping

from collections import OrderedDict, defaultdict

__all__ = ['Mapping', 'MutableMapping', 'OrderedDict', 'defaultdict']

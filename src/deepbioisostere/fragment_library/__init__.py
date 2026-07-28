"""Fragment-library preprocessing.

``parse_fragments`` turns ``fragment_library.csv`` into the tensor caches
(``frag_features.pkl``, ``frag_brics_maskings.pkl``) that training and
generation consume. It used to live outside the importable package and was
reached via ``sys.path`` manipulation, which broke as soon as the project was
installed rather than run from a checkout.
"""

from __future__ import annotations

__all__ = ["FragLibProcessor"]


def __getattr__(name: str):
    if name == "FragLibProcessor":
        from .parse_fragments import FragLibProcessor

        return FragLibProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""DeepBioisostere: autonomous bioisosteric replacement for multi-property optimization.

Reference implementation for:

    Kim, H.*, Moon, S.*, Zhung, W., Kim, S., Lim, J. & Kim, W. Y.
    "Autonomous bioisosteric replacement for multi-property optimization in
    drug design." Nature Communications (2026).
    https://doi.org/10.1038/s41467-026-75512-9

Typical use::

    from deepbioisostere import Conditioner, DeepBioisostere, Generator

    model = DeepBioisostere.from_pretrained(properties=["mw", "logp"])
    generator = Generator(
        model=model,
        conditioner=Conditioner(phase="generation", properties=["logp", "mw"]),
        properties=["logp", "mw"],
    )
    df = generator.generate([(smiles, {"mw": 0, "logp": -1})])

Attribute access is lazy (PEP 562): importing this package does not pull in
torch or rdkit, and never touches the network. That cost is paid on first use
of a symbol instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "AssetError",
    "BaselineGenerator",
    "Conditioner",
    "DeepBioisostere",
    "Generator",
    "PROPERTIES",
    "calc_Mw",
    "calc_QED",
    "calc_SAscore",
    "calc_logP",
    "resolve_checkpoint",
    "resolve_fragment_library",
    "__version__",
]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("deepbioisostere")
except PackageNotFoundError:  # pragma: no cover - checkout without an install
    __version__ = "0.0.0+unknown"

# Symbol -> submodule that defines it.
_LAZY_EXPORTS = {
    "AssetError": "assets",
    "resolve_checkpoint": "assets",
    "resolve_fragment_library": "assets",
    "BaselineGenerator": "baseline_generator",
    "Conditioner": "conditioning",
    "DeepBioisostere": "model",
    "Generator": "generate",
    "PROPERTIES": "property",
    "calc_Mw": "property",
    "calc_QED": "property",
    "calc_SAscore": "property",
    "calc_logP": "property",
}

if TYPE_CHECKING:  # give type checkers and IDEs the real symbols
    from .assets import AssetError, resolve_checkpoint, resolve_fragment_library
    from .baseline_generator import BaselineGenerator
    from .conditioning import Conditioner
    from .generate import Generator
    from .model import DeepBioisostere
    from .property import PROPERTIES, calc_logP, calc_Mw, calc_QED, calc_SAscore


def __getattr__(name: str):
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache so the lazy path runs at most once
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

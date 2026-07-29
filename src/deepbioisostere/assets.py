"""On-demand resolution of model checkpoints and the fragment library.

Checkpoints (~20 MB each) and the fragment library (~13 MB, plus derived tensor
caches that are considerably larger) are too big to ship inside a wheel. They
live in a Hugging Face Hub repository and are fetched on first use.

Resolution order for every asset, first hit wins:

1. an explicit path passed by the caller,
2. ``$DEEPBIOISOSTERE_ASSET_DIR``,
3. the Hugging Face Hub.

That ordering means an existing source checkout with ``model_save/`` and
``fragment_library/`` in place keeps working unchanged, offline and air-gapped
installs stay possible, and only a fresh install reaches the network. Nothing
here downloads at import time.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "AssetError",
    "AVAILABLE_PROPERTY_SETS",
    "AVAILABLE_ABLATION_SETS",
    "VALID_PROPERTIES",
    "checkpoint_filename",
    "default_cache_dir",
    "hf_repo_id",
    "resolve_checkpoint",
    "resolve_fragment_library",
]

# Asset resolution is the one part of a first run that is slow and invisible:
# `generate` can sit for minutes fetching a 712 MB tensor cache with nothing on
# screen to say so. These messages explain the wait and, just as usefully, say
# *which* copy of an asset was chosen -- silently picking up a different local
# fragment library is the failure that silently changes results.
#
# Library convention: log, never configure. Nothing is printed unless the
# application attaches a handler, so importing this module stays quiet. The CLI
# turns it on; `--quiet` turns it back off.
logger = logging.getLogger(__name__)


def _describe_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "unknown size"
    return f"{size / 1e6:.1f} MB"

# Overridable at runtime with $DEEPBIOISOSTERE_HF_REPO.
DEFAULT_HF_REPO_ID = "mseok/DeepBioisostere"

FRAGMENT_LIBRARY_CSV = "fragment_library.csv"
# Derived tensor caches. Absent from the Hub repo they are regenerated locally
# from the csv, which costs a few minutes of CPU on first use.
FRAGMENT_LIBRARY_DERIVED = ("frag_features.pkl", "frag_brics_maskings.pkl")

VALID_PROPERTIES = ("logp", "mw", "qed", "sa")

# Property sets with a published checkpoint. Keys are the sorted tuple of
# property names; note the pair coverage is partial by design -- the paper
# trains three of the six possible pairs.
AVAILABLE_PROPERTY_SETS: tuple[tuple[str, ...], ...] = (
    ("logp",),
    ("mw",),
    ("qed",),
    ("sa",),
    ("logp", "mw"),
    ("mw", "qed"),
    ("qed", "sa"),
)

# Ablation checkpoints (use_subgraph_AMPN=False) exist only for the pairs.
AVAILABLE_ABLATION_SETS: tuple[tuple[str, ...], ...] = (
    ("logp", "mw"),
    ("mw", "qed"),
    ("qed", "sa"),
)


class AssetError(RuntimeError):
    """Raised when an asset cannot be resolved locally or from the Hub."""


def hf_repo_id() -> str:
    """The Hub repo to fetch from, overridable per-run."""
    return os.environ.get("DEEPBIOISOSTERE_HF_REPO", DEFAULT_HF_REPO_ID)


def default_cache_dir() -> Path:
    """Writable directory for downloaded and derived assets."""
    if env := os.environ.get("DEEPBIOISOSTERE_CACHE_DIR"):
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "deepbioisostere"


def _asset_dir_override() -> Path | None:
    if env := os.environ.get("DEEPBIOISOSTERE_ASSET_DIR"):
        return Path(env).expanduser()
    return None


def _normalize_properties(properties: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(sorted(p.lower() for p in properties))
    unknown = [p for p in normalized if p not in VALID_PROPERTIES]
    if unknown:
        raise AssetError(
            f"Unknown propert{'y' if len(unknown) == 1 else 'ies'}: "
            f"{', '.join(unknown)}. Valid properties are: "
            f"{', '.join(VALID_PROPERTIES)}."
        )
    if not normalized:
        raise AssetError(
            "At least one property is required to select a checkpoint. "
            f"Valid properties are: {', '.join(VALID_PROPERTIES)}."
        )
    return normalized


def checkpoint_filename(properties: Sequence[str], ablation: bool = False) -> str:
    """Filename for a checkpoint, e.g. ``DeepBioisostere_logp_mw.pt``.

    Property names are sorted: the published files are ``logp_mw``, never
    ``mw_logp``.
    """
    normalized = _normalize_properties(properties)
    suffix = "_ablation" if ablation else ""
    return f"DeepBioisostere_{'_'.join(normalized)}{suffix}.pt"


def _check_availability(normalized: tuple[str, ...], ablation: bool) -> None:
    available = AVAILABLE_ABLATION_SETS if ablation else AVAILABLE_PROPERTY_SETS
    if normalized in available:
        return
    kind = "ablation checkpoint" if ablation else "checkpoint"
    listed = "\n".join("  - " + ", ".join(s) for s in available)
    raise AssetError(
        f"No {kind} was published for the property set "
        f"({', '.join(normalized)}). Available {kind} property sets are:\n"
        f"{listed}"
    )


def _download(filename: str, local_dir: Path) -> Path:
    repo_id = hf_repo_id()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise AssetError(
            "huggingface_hub is required to download assets. Install it with "
            "`pip install huggingface_hub`, or supply the assets locally via "
            "$DEEPBIOISOSTERE_ASSET_DIR."
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "downloading %s from Hugging Face repo %s into %s", filename, repo_id,
        local_dir,
    )
    started = time.monotonic()
    try:
        # local_dir gives real files rather than symlinks into the blob store,
        # which matters because the fragment library directory is written to.
        path = hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=str(local_dir)
        )
    except Exception as exc:
        raise AssetError(
            f"Failed to download {filename!r} from Hugging Face repo "
            f"{repo_id!r}: {exc}\n"
            "If you are offline or behind a firewall, point "
            "$DEEPBIOISOSTERE_ASSET_DIR at a directory that already holds the "
            "assets (for example the model_save/ directory of a source "
            "checkout), or set $DEEPBIOISOSTERE_HF_REPO to a mirror."
        ) from exc
    resolved = Path(path)
    logger.info(
        "downloaded %s (%s) in %.1fs", filename, _describe_size(resolved),
        time.monotonic() - started,
    )
    return resolved


def resolve_checkpoint(
    properties: Sequence[str],
    ablation: bool = False,
    local_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return a local path to the checkpoint for ``properties``.

    Downloads it on first use unless it is already available locally.
    """
    normalized = _normalize_properties(properties)
    _check_availability(normalized, ablation)
    filename = checkpoint_filename(normalized, ablation)

    # An explicit directory is exclusive, not merely first in line. Falling
    # through to $DEEPBIOISOSTERE_ASSET_DIR or the Hub would quietly hand back
    # a *different* checkpoint than the one that was asked for.
    if local_dir is not None:
        explicit = Path(local_dir).expanduser()
        candidate = explicit / filename
        if candidate.is_file():
            logger.info("checkpoint %s: using explicit path %s (%s)", filename,
                        candidate, _describe_size(candidate))
            return candidate
        raise AssetError(
            f"{filename!r} was not found in {explicit}. "
            "Omit the local path to fetch it from the Hugging Face Hub."
        )

    if (override := _asset_dir_override()) is not None:
        for directory in (override, override / "model_save"):
            candidate = directory / filename
            if candidate.is_file():
                logger.info(
                    "checkpoint %s: using $DEEPBIOISOSTERE_ASSET_DIR copy %s (%s)",
                    filename, candidate, _describe_size(candidate),
                )
                return candidate

    cached = default_cache_dir() / "checkpoints"
    if (candidate := cached / filename).is_file():
        logger.info("checkpoint %s: cache hit at %s (%s)", filename, candidate,
                    _describe_size(candidate))
        return candidate
    logger.info("checkpoint %s: not cached, fetching", filename)
    return _download(filename, cached)


def resolve_fragment_library(
    local_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return a local *directory* holding the fragment library.

    The directory is what the rest of the package consumes: it must contain
    ``fragment_library.csv``, and it must be writable, because the derived
    tensor caches are generated into it on first use if they are absent.
    """
    # As in resolve_checkpoint, an explicit directory is exclusive.
    if local_dir is not None:
        explicit = Path(local_dir).expanduser()
        if (explicit / FRAGMENT_LIBRARY_CSV).is_file():
            _log_library(explicit, "explicit path")
            return explicit
        raise AssetError(
            f"{FRAGMENT_LIBRARY_CSV!r} was not found in {explicit}. "
            "Omit the local path to fetch the fragment library from the "
            "Hugging Face Hub."
        )

    if (override := _asset_dir_override()) is not None:
        for directory in (override, override / "fragment_library"):
            if (directory / FRAGMENT_LIBRARY_CSV).is_file():
                _log_library(directory, "$DEEPBIOISOSTERE_ASSET_DIR")
                return directory

    cached = default_cache_dir() / "fragment_library"
    cold = not (cached / FRAGMENT_LIBRARY_CSV).is_file()
    if cold:
        logger.info("fragment library: not cached, fetching")
        _download(FRAGMENT_LIBRARY_CSV, cached)

    # Derived caches are a nice-to-have: pull them if the Hub publishes them,
    # otherwise they are regenerated locally from the csv.
    #
    # Only on a cold cache. frag_brics_maskings.pkl is deliberately NOT on the
    # Hub -- it is 2.9 GB and training-only -- so retrying it on a warm cache
    # meant a failed network round-trip on every single `generate`, plus a
    # confusing "not available" line each time.
    if cold:
        for derived in FRAGMENT_LIBRARY_DERIVED:
            if (cached / derived).is_file():
                continue
            try:
                _download(derived, cached)
            except AssetError:
                logger.info(
                    "fragment library: %s is not distributed on the Hub. "
                    "Generation does not need it; training does, and it is "
                    "built locally on demand (about an hour) or downloaded "
                    "from the Zenodo record.", derived,
                )
    _log_library(cached, "cache")
    return cached


def _log_library(directory: Path, source: str) -> None:
    """Report which library was chosen and which derived caches it already has.

    Which fragment library is in play decides which molecules get generated --
    insertion fragments are selected by index into it -- so this is the single
    most consequential resolution the package makes.
    """
    if not logger.isEnabledFor(logging.INFO):
        return
    present = [d for d in FRAGMENT_LIBRARY_DERIVED if (directory / d).is_file()]
    missing = [d for d in FRAGMENT_LIBRARY_DERIVED if d not in present]
    logger.info("fragment library: using %s (%s)", directory, source)
    csv = directory / FRAGMENT_LIBRARY_CSV
    logger.info("    %s (%s)", FRAGMENT_LIBRARY_CSV, _describe_size(csv))
    for name in present:
        logger.info("    %s (%s)", name, _describe_size(directory / name))
    for name in missing:
        logger.info("    %s absent (built on demand)", name)

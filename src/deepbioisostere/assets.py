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

import os
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

# Replace with the real namespace once the Hub repository is created, or set
# $DEEPBIOISOSTERE_HF_REPO at runtime.
DEFAULT_HF_REPO_ID = "REPLACE_ME/DeepBioisostere"

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
    if repo_id.startswith("REPLACE_ME/"):
        raise AssetError(
            f"Cannot download {filename!r}: the Hugging Face repository for "
            "DeepBioisostere assets has not been configured.\n"
            "Either set $DEEPBIOISOSTERE_HF_REPO to the Hub repo id, or point "
            "$DEEPBIOISOSTERE_ASSET_DIR at a directory that already contains "
            "the assets (for example the model_save/ directory of a source "
            "checkout)."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise AssetError(
            "huggingface_hub is required to download assets. Install it with "
            "`pip install huggingface_hub`, or supply the assets locally via "
            "$DEEPBIOISOSTERE_ASSET_DIR."
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        # local_dir gives real files rather than symlinks into the blob store,
        # which matters because the fragment library directory is written to.
        path = hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=str(local_dir)
        )
    except Exception as exc:
        raise AssetError(
            f"Failed to download {filename!r} from Hugging Face repo {repo_id!r}: {exc}"
        ) from exc
    return Path(path)


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

    search_dirs: list[Path] = []
    if local_dir is not None:
        search_dirs.append(Path(local_dir).expanduser())
    if (override := _asset_dir_override()) is not None:
        search_dirs.append(override)
        search_dirs.append(override / "model_save")
    for directory in search_dirs:
        candidate = directory / filename
        if candidate.is_file():
            return candidate

    if local_dir is not None:
        # An explicit directory was given but does not hold the file. Say so
        # rather than silently reaching for the network.
        raise AssetError(
            f"{filename!r} was not found in {Path(local_dir).expanduser()}. "
            "Omit the local path to fetch it from the Hugging Face Hub."
        )

    cached = default_cache_dir() / "checkpoints"
    if (candidate := cached / filename).is_file():
        return candidate
    return _download(filename, cached)


def resolve_fragment_library(
    local_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return a local *directory* holding the fragment library.

    The directory is what the rest of the package consumes: it must contain
    ``fragment_library.csv``, and it must be writable, because the derived
    tensor caches are generated into it on first use if they are absent.
    """
    search_dirs: list[Path] = []
    if local_dir is not None:
        search_dirs.append(Path(local_dir).expanduser())
    if (override := _asset_dir_override()) is not None:
        search_dirs.append(override)
        search_dirs.append(override / "fragment_library")
    for directory in search_dirs:
        if (directory / FRAGMENT_LIBRARY_CSV).is_file():
            return directory

    if local_dir is not None:
        raise AssetError(
            f"{FRAGMENT_LIBRARY_CSV!r} was not found in "
            f"{Path(local_dir).expanduser()}. Omit the local path to fetch the "
            "fragment library from the Hugging Face Hub."
        )

    cached = default_cache_dir() / "fragment_library"
    if not (cached / FRAGMENT_LIBRARY_CSV).is_file():
        _download(FRAGMENT_LIBRARY_CSV, cached)
    # Derived caches are a nice-to-have: pull them if the Hub repo publishes
    # them, otherwise they are regenerated locally from the csv.
    for derived in FRAGMENT_LIBRARY_DERIVED:
        if not (cached / derived).is_file():
            try:
                _download(derived, cached)
            except AssetError:
                pass
    return cached

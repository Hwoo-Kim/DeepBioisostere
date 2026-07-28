"""The published fragment-library caches must stay loadable after the refactor.

`frag_features.pkl` as published on the Hub is the artifact behind the paper's
numbers, pickled when this code was a flat set of top-level modules. It names
its values `data.PairData`, a module path that no longer exists now that the
package is `deepbioisostere.*`.

Rewriting the file would break its provenance, so the rename is absorbed on
read. Without that, every user who downloads the cache hits

    AttributeError: Can't get attribute 'PairData' on <module 'data' ...>

on their very first `generate` call. These tests pin that behaviour.
"""

from __future__ import annotations

import io
import pickle
import sys
import types

import pytest

from deepbioisostere.data import PairData
from deepbioisostere.dataset import _load_frag_cache, _LegacyUnpickler


def _unpickler(blob: bytes = b"") -> _LegacyUnpickler:
    return _LegacyUnpickler(io.BytesIO(blob))


@pytest.fixture
def legacy_blob():
    """A pickle whose classes are named under the pre-package module layout.

    `pickle` writes the class's own ``__module__``, so registering a fake
    ``data`` module is not enough on its own: the blob would still say
    ``deepbioisostere.data`` and the test would pass vacuously. The class
    attribute has to be repointed for the duration of the dump, which is what
    makes this byte-for-byte the shape of the published cache.
    """
    fake = types.ModuleType("data")
    fake.PairData = PairData
    sys.modules["data"] = fake
    original_module = PairData.__module__
    PairData.__module__ = "data"
    try:
        blob = pickle.dumps({"[16*]c1ccccc1": PairData()})
    finally:
        PairData.__module__ = original_module
        sys.modules.pop("data", None)
    assert b"data\x94\x8c\x08PairData" in blob or b"\x8c\x04data" in blob
    yield blob


class TestModuleRemapping:
    def test_legacy_data_module_resolves(self):
        assert _unpickler().find_class("data", "PairData") is PairData

    def test_scripts_prefixed_module_resolves(self):
        assert _unpickler().find_class("scripts.data", "PairData") is PairData

    def test_current_module_is_unchanged(self):
        found = _unpickler().find_class("deepbioisostere.data", "PairData")
        assert found is PairData

    def test_unrelated_modules_are_not_rewritten(self):
        """The remap is a fixed table, not a blanket 'prepend the package'."""
        assert _unpickler().find_class("collections", "OrderedDict") is not None


class TestLoading:
    def test_legacy_pickle_loads(self, legacy_blob):
        loaded = _unpickler(legacy_blob).load()
        assert isinstance(next(iter(loaded.values())), PairData)

    def test_plain_unpickler_fails_on_the_same_blob(self, legacy_blob):
        """Guards the test itself: the fixture must really exercise the remap."""
        with pytest.raises(AttributeError):
            pickle.loads(legacy_blob)

    def test_load_frag_cache_reads_from_disk(self, legacy_blob, tmp_path):
        path = tmp_path / "frag_features.pkl"
        path.write_bytes(legacy_blob)
        loaded = _load_frag_cache(path)
        assert isinstance(next(iter(loaded.values())), PairData)

    def test_caches_written_by_this_package_still_load(self, tmp_path):
        """Round-trip under the current module names, the forward-looking case."""
        path = tmp_path / "frag_features.pkl"
        path.write_bytes(pickle.dumps({"[16*]c1ccccc1": PairData()}))
        loaded = _load_frag_cache(path)
        assert isinstance(next(iter(loaded.values())), PairData)

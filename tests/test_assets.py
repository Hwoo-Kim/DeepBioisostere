"""Tests for asset resolution.

These cover the precedence chain and, importantly, the error paths -- the
unavailable-property-set case must fail with an explanation rather than a 404
from the Hub, and the unconfigured-repo case must say what to do about it.
"""

from __future__ import annotations

import pytest

from deepbioisostere.assets import (
    AVAILABLE_ABLATION_SETS,
    AVAILABLE_PROPERTY_SETS,
    AssetError,
    checkpoint_filename,
    default_cache_dir,
    hf_repo_id,
    resolve_checkpoint,
    resolve_fragment_library,
)


class TestCheckpointFilename:
    def test_properties_are_sorted(self):
        # The published files are logp_mw, never mw_logp.
        assert checkpoint_filename(["mw", "logp"]) == "DeepBioisostere_logp_mw.pt"
        assert checkpoint_filename(["logp", "mw"]) == "DeepBioisostere_logp_mw.pt"

    def test_case_is_normalized(self):
        assert checkpoint_filename(["MW"]) == "DeepBioisostere_mw.pt"

    def test_ablation_suffix(self):
        assert (
            checkpoint_filename(["qed", "sa"], ablation=True)
            == "DeepBioisostere_qed_sa_ablation.pt"
        )

    def test_unknown_property_is_rejected(self):
        with pytest.raises(AssetError, match="Unknown propert"):
            checkpoint_filename(["tpsa"])

    def test_empty_property_set_is_rejected(self):
        with pytest.raises(AssetError, match="At least one property"):
            checkpoint_filename([])


class TestAvailability:
    def test_unpublished_pair_explains_rather_than_404s(self):
        # logp+qed was never trained; the error must say so up front.
        with pytest.raises(AssetError) as exc:
            resolve_checkpoint(["logp", "qed"])
        assert "No checkpoint was published" in str(exc.value)
        assert "logp, mw" in str(exc.value)  # lists what IS available

    def test_ablation_only_exists_for_pairs(self):
        with pytest.raises(AssetError, match="No ablation checkpoint"):
            resolve_checkpoint(["mw"], ablation=True)

    def test_every_advertised_set_has_a_wellformed_filename(self):
        for combo in AVAILABLE_PROPERTY_SETS:
            assert checkpoint_filename(combo).startswith("DeepBioisostere_")
        for combo in AVAILABLE_ABLATION_SETS:
            assert checkpoint_filename(combo, ablation=True).endswith("_ablation.pt")


class TestResolution:
    def test_explicit_local_dir_wins(self, tmp_path):
        target = tmp_path / "DeepBioisostere_mw.pt"
        target.write_bytes(b"stub")
        assert resolve_checkpoint(["mw"], local_dir=tmp_path) == target

    def test_explicit_local_dir_miss_does_not_hit_network(self, tmp_path):
        # An explicit path that misses is a user error; say so instead of
        # silently downloading something different.
        with pytest.raises(AssetError, match="was not found in"):
            resolve_checkpoint(["mw"], local_dir=tmp_path)

    def test_explicit_local_dir_does_not_fall_through_to_env(
        self, tmp_path, monkeypatch
    ):
        """An explicit directory is exclusive, not merely first in line.

        Regression test. Previously both the explicit directory and
        $DEEPBIOISOSTERE_ASSET_DIR were searched, so an explicit path that
        missed would silently return a *different* checkpoint from the env
        directory. That only shows up when the env var happens to be set, which
        is why it survived a local run and failed on a compute node.
        """
        decoy = tmp_path / "decoy"
        (decoy / "model_save").mkdir(parents=True)
        (decoy / "model_save" / "DeepBioisostere_mw.pt").write_bytes(b"decoy")
        monkeypatch.setenv("DEEPBIOISOSTERE_ASSET_DIR", str(decoy))

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(AssetError, match="was not found in"):
            resolve_checkpoint(["mw"], local_dir=empty)

    def test_explicit_frag_lib_dir_does_not_fall_through_to_env(
        self, tmp_path, monkeypatch
    ):
        decoy = tmp_path / "decoy" / "fragment_library"
        decoy.mkdir(parents=True)
        (decoy / "fragment_library.csv").write_text("INDEX\tFRAG-SMI\n")
        monkeypatch.setenv("DEEPBIOISOSTERE_ASSET_DIR", str(tmp_path / "decoy"))

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(AssetError, match="was not found in"):
            resolve_fragment_library(local_dir=empty)

    def test_asset_dir_env_is_used(self, tmp_path, monkeypatch):
        (tmp_path / "model_save").mkdir()
        target = tmp_path / "model_save" / "DeepBioisostere_qed.pt"
        target.write_bytes(b"stub")
        monkeypatch.setenv("DEEPBIOISOSTERE_ASSET_DIR", str(tmp_path))
        assert resolve_checkpoint(["qed"]) == target

    def test_asset_dir_env_flat_layout_also_works(self, tmp_path, monkeypatch):
        target = tmp_path / "DeepBioisostere_sa.pt"
        target.write_bytes(b"stub")
        monkeypatch.setenv("DEEPBIOISOSTERE_ASSET_DIR", str(tmp_path))
        assert resolve_checkpoint(["sa"]) == target

    def test_fragment_library_dir_from_env(self, tmp_path, monkeypatch):
        lib = tmp_path / "fragment_library"
        lib.mkdir()
        (lib / "fragment_library.csv").write_text("INDEX\tFRAG-SMI\n")
        monkeypatch.setenv("DEEPBIOISOSTERE_ASSET_DIR", str(tmp_path))
        assert resolve_fragment_library() == lib

    def test_unconfigured_repo_gives_actionable_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEEPBIOISOSTERE_ASSET_DIR", raising=False)
        monkeypatch.delenv("DEEPBIOISOSTERE_HF_REPO", raising=False)
        monkeypatch.setenv("DEEPBIOISOSTERE_CACHE_DIR", str(tmp_path))
        with pytest.raises(AssetError) as exc:
            resolve_checkpoint(["mw"])
        message = str(exc.value)
        assert "has not been configured" in message
        assert "DEEPBIOISOSTERE_HF_REPO" in message
        assert "DEEPBIOISOSTERE_ASSET_DIR" in message


class TestConfiguration:
    def test_repo_id_is_overridable(self, monkeypatch):
        monkeypatch.setenv("DEEPBIOISOSTERE_HF_REPO", "someone/else")
        assert hf_repo_id() == "someone/else"

    def test_cache_dir_is_overridable(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DEEPBIOISOSTERE_CACHE_DIR", str(tmp_path))
        assert default_cache_dir() == tmp_path

    def test_cache_dir_respects_xdg(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DEEPBIOISOSTERE_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert default_cache_dir() == tmp_path / "deepbioisostere"

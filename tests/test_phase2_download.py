"""TDD tests for Phase 2 dataset download manifest (ADR 0001 §3.2).

Dry-run only — does NOT hit network. Real downloads happen later when GPU is free.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyberpuppy.data.phase2_download import (DatasetSpec, DownloadManifest,
                                              Phase2Downloader,
                                              load_default_specs)

pytestmark = pytest.mark.unit


# ---- 1. Default specs ----------------------------------------------------

def test_default_specs_present() -> None:
    specs = load_default_specs()
    names = {s.name for s in specs}
    # Must list at minimum the four datasets ADR §3.2 commits to
    assert {"cold", "sccd", "chnci", "state_toxicn"}.issubset(names)


def test_each_spec_has_required_fields() -> None:
    for s in load_default_specs():
        assert s.name and s.kind in {"hf_dataset", "git", "url"}
        assert s.source  # HF id, git url, or http url
        assert isinstance(s.license, str) and s.license


def test_cold_spec_uses_hf_id() -> None:
    spec = next(s for s in load_default_specs() if s.name == "cold")
    assert spec.kind == "hf_dataset"
    assert spec.source == "thu-coai/cold"
    assert "apache" in spec.license.lower()


# ---- 2. Manifest model ---------------------------------------------------

def test_manifest_init_empty() -> None:
    m = DownloadManifest()
    assert m.entries == []
    assert m.to_dict()["entries"] == []


def test_manifest_add_and_serialize() -> None:
    m = DownloadManifest()
    m.add(name="cold", kind="hf_dataset", source="thu-coai/cold",
          license="apache-2.0", status="planned", target_path="data/raw/cold")
    d = m.to_dict()
    assert len(d["entries"]) == 1
    assert d["entries"][0]["name"] == "cold"
    assert d["entries"][0]["status"] == "planned"


def test_manifest_round_trip(tmp_path: Path) -> None:
    m = DownloadManifest()
    m.add(name="x", kind="git", source="https://github.com/foo/bar",
          license="mit", status="planned", target_path="data/raw/x")
    out = tmp_path / "manifest.json"
    m.write(out)
    assert out.exists()
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["entries"][0]["source"] == "https://github.com/foo/bar"


# ---- 3. Downloader dry-run ------------------------------------------------

@pytest.fixture
def downloader(tmp_path: Path) -> Phase2Downloader:
    return Phase2Downloader(target_dir=tmp_path / "data" / "raw", dry_run=True)


def test_downloader_no_network_in_dry_run(downloader: Phase2Downloader) -> None:
    spec = DatasetSpec(name="cold", kind="hf_dataset",
                       source="thu-coai/cold", license="apache-2.0")
    result = downloader.fetch(spec)
    assert result["status"] == "planned"
    assert result["target_path"].endswith("/data/raw/cold")
    # No directory should be created in dry-run
    assert not Path(result["target_path"]).exists()


def test_downloader_writes_manifest(downloader: Phase2Downloader, tmp_path: Path) -> None:
    specs = [
        DatasetSpec(name="cold", kind="hf_dataset",
                    source="thu-coai/cold", license="apache-2.0"),
        DatasetSpec(name="state_toxicn", kind="git",
                    source="https://github.com/shenmeyemeifashengguo/STATE-ToxiCN",
                    license="academic"),
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest = downloader.fetch_all(specs, manifest_path=manifest_path)
    assert manifest_path.exists()
    assert len(manifest.entries) == 2
    assert all(e["status"] == "planned" for e in manifest.to_dict()["entries"])


def test_downloader_supports_url_kind(downloader: Phase2Downloader) -> None:
    spec = DatasetSpec(name="something", kind="url",
                       source="https://example.com/data.zip", license="public")
    result = downloader.fetch(spec)
    assert result["status"] == "planned"


def test_downloader_rejects_unknown_kind(downloader: Phase2Downloader) -> None:
    spec = DatasetSpec(name="bad", kind="ftp",  # type: ignore[arg-type]
                       source="ftp://example.com", license="public")
    with pytest.raises(ValueError, match="kind"):
        downloader.fetch(spec)


# ---- 4. Live mode (network) is OFF in tests ------------------------------

def test_live_mode_requires_explicit_opt_in(tmp_path: Path) -> None:
    """Default constructor must default to dry_run=True so tests / CI never download."""
    d = Phase2Downloader(target_dir=tmp_path)
    assert d.dry_run is True

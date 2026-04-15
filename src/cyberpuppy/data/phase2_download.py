"""Phase 2 dataset downloader (ADR 0001 §3.2).

Dry-run by default — never hits network unless explicitly opted in.
Designed so CI / tests / GPU-paused sessions can plan & verify intent
without consuming bandwidth or VRAM.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

DatasetKind = Literal["hf_dataset", "git", "url"]


@dataclass
class DatasetSpec:
    name: str
    kind: DatasetKind
    source: str
    license: str
    notes: str = ""


@dataclass
class DownloadManifest:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        *,
        name: str,
        kind: str,
        source: str,
        license: str,
        status: str,
        target_path: str,
    ) -> None:
        self.entries.append({
            "name": name,
            "kind": kind,
            "source": source,
            "license": license,
            "status": status,
            "target_path": target_path,
        })

    def to_dict(self) -> dict[str, Any]:
        return {"entries": list(self.entries)}

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_default_specs() -> list[DatasetSpec]:
    """Datasets named in ADR 0001 §3.2."""
    return [
        DatasetSpec(
            name="cold",
            kind="hf_dataset",
            source="thu-coai/cold",
            license="apache-2.0",
            notes="既有 baseline；HF 直接 load_dataset",
        ),
        DatasetSpec(
            name="sccd",
            kind="git",
            source="https://github.com/yangxinyu/SCCD",  # 待人工確認
            license="academic-pending-confirmation",
            notes="arXiv 2501.15042; GitHub URL 待 paper 作者頁確認",
        ),
        DatasetSpec(
            name="chnci",
            kind="git",
            source="https://github.com/CHNCI/CHNCI",  # 待人工確認
            license="academic-pending-confirmation",
            notes="arXiv 2505.20654; GitHub URL 待 paper 作者頁確認",
        ),
        DatasetSpec(
            name="state_toxicn",
            kind="git",
            source="https://github.com/shenmeyemeifashengguo/STATE-ToxiCN",
            license="academic",
            notes="ACL Findings 2025; 含 830 詞中文仇恨俚語詞典",
        ),
        DatasetSpec(
            name="toxicloak_cn",
            kind="git",
            source="https://github.com/DUT-lujunyu/ToxiCloakCN",  # 待人工確認
            license="academic-pending-confirmation",
            notes="EMNLP 2024；僅作 robustness 評測，不訓練",
        ),
    ]


class Phase2Downloader:
    def __init__(
        self,
        target_dir: Path | str,
        dry_run: bool = True,
    ) -> None:
        self.target_dir = Path(target_dir)
        self.dry_run = dry_run
        if not dry_run:
            logger.warning(
                "Phase2Downloader live mode enabled — will hit network. "
                "Confirm GPU availability and bandwidth budget."
            )

    def _target_path(self, spec: DatasetSpec) -> Path:
        return self.target_dir / spec.name

    def fetch(self, spec: DatasetSpec) -> dict[str, Any]:
        if spec.kind not in {"hf_dataset", "git", "url"}:
            raise ValueError(f"Unsupported kind: {spec.kind!r}")

        target = self._target_path(spec)
        if self.dry_run:
            return {
                "name": spec.name,
                "kind": spec.kind,
                "source": spec.source,
                "license": spec.license,
                "status": "planned",
                "target_path": str(target),
            }

        # Live mode — kept lean; real implementations land when GPU free.
        target.mkdir(parents=True, exist_ok=True)
        if spec.kind == "hf_dataset":
            from datasets import load_dataset  # type: ignore[import-not-found]
            load_dataset(spec.source, cache_dir=str(target))
        elif spec.kind == "git":
            import subprocess
            subprocess.run(
                ["git", "clone", "--depth", "1", spec.source, str(target)],
                check=True,
            )
        elif spec.kind == "url":
            import urllib.request
            urllib.request.urlretrieve(spec.source, target / Path(spec.source).name)

        return {
            "name": spec.name,
            "kind": spec.kind,
            "source": spec.source,
            "license": spec.license,
            "status": "fetched",
            "target_path": str(target),
        }

    def fetch_all(
        self,
        specs: list[DatasetSpec],
        manifest_path: Path | None = None,
    ) -> DownloadManifest:
        manifest = DownloadManifest()
        for spec in specs:
            result = self.fetch(spec)
            manifest.add(
                name=result["name"],
                kind=result["kind"],
                source=result["source"],
                license=result["license"],
                status=result["status"],
                target_path=result["target_path"],
            )
        if manifest_path is not None:
            manifest.write(manifest_path)
        return manifest

"""ToxiCloakCN-style homophone augmenter for v2.4 training (in-distribution synth).

v2.3 failure recap: rule-based pypinyin substitution on a 829-term slang
lexicon produced data that did NOT match ToxiCloakCN heldout's distribution
(homo drop went from −8.51% to −9.23%, worse). Per ADR Phase 3.4, the root
cause is "synthetic data must match test distribution to be useful".

v2.4 fix: use `jionlp.homophone_substitution` — the EXACT library
ToxiCloakCN's authors used per paper §3.1.2 ("Keywords were identified as
those present in the specified dictionary" = JioNLP). This makes
distribution match by construction, not by hope.

Pipeline per call:
  Traditional text  --(OpenCC t2s)-->  Simplified
                                   --(JioNLP homophone_substitution)-->  Simp variants
                                   --(OpenCC s2twp)-->  Traditional output
"""
from __future__ import annotations

from typing import Any, List, Optional

from .phase2 import UnifiedRecord


class ToxiCloakStyleAugmenter:
    """JioNLP-backed homophone augmenter (matches ToxiCloakCN distribution).

    Parameters
    ----------
    homo_ratio : float
        Per-word substitution probability passed to JioNLP. ToxiCloakCN
        paper §3.1.2 used 0.3 (30% character-level perturbation).
    max_variants : int
        How many candidate variants to ask JioNLP for (we keep the first
        one that actually differs from input).
    allow_mispronounce : bool
        Allow JioNLP's dialect-confusion substitutions (zh/z, eng/en, l/n).
        ToxiCloakCN-style attacks include these so default True.
    seed : int
        RNG seed for reproducibility (0 = nondeterministic). Default 42.
    opencc_t2s_conf, opencc_s2t_conf : str
        OpenCC configs for round-tripping Traditional <-> Simplified.
        Default `t2s` and `s2twp` (Taiwan-style with phrase mapping).
    """

    def __init__(
        self,
        homo_ratio: float = 0.3,
        max_variants: int = 3,
        allow_mispronounce: bool = True,
        seed: int = 42,
        opencc_t2s_conf: str = "t2s",
        opencc_s2t_conf: str = "s2twp",
    ) -> None:
        import jionlp as jio
        import numpy as np
        from opencc import OpenCC

        # Use a FRESH JioNLP HomophoneSubstitution instance per augmenter
        # so other code paths can't pollute its internal state. JioNLP's
        # public `jio.homophone_substitution` is a module-level singleton —
        # sharing it across augmenters breaks determinism.
        from jionlp.textaug.homophone_substitution import HomophoneSubstitution

        self._np = np
        self._hs = HomophoneSubstitution()
        self._t2s = OpenCC(opencc_t2s_conf)
        self._s2t = OpenCC(opencc_s2t_conf)
        self.homo_ratio = float(homo_ratio)
        self.max_variants = int(max_variants)
        self.allow_mispronounce = bool(allow_mispronounce)
        self.seed = int(seed)

    def augment_text(self, text: str) -> Optional[str]:
        """Return one Traditional-Chinese augmented variant, or None.

        Returns None when:
          - input is empty
          - JioNLP returns no variants (e.g. no Chinese in text)
          - all returned variants round-trip identical to the input
        """
        if not text or not text.strip():
            return None

        simp = self._t2s.convert(text)
        # Force-reset numpy global RNG before each call. JioNLP's `seed`
        # param only re-seeds when the value CHANGES between calls, so
        # consecutive calls with the same seed accumulate state. Resetting
        # numpy's RNG explicitly guarantees per-call reproducibility.
        self._np.random.seed(self.seed)
        try:
            variants = self._hs(
                simp,
                augmentation_num=self.max_variants,
                homo_ratio=self.homo_ratio,
                allow_mispronounce=self.allow_mispronounce,
                seed=self.seed,
            )
        except Exception:
            return None

        if not variants:
            return None

        for v in variants:
            v_trad = self._s2t.convert(v)
            if v_trad and v_trad != text:
                return v_trad
        return None

    def build_pair_records(
        self,
        src_record: dict[str, Any],
        pair_id: int,
    ) -> List[UnifiedRecord]:
        """Produce a (base, homo) pair sharing `cloak_pair_id`.

        Returns [] when augmentation fails — caller should advance `pair_id`
        without consuming it (or just skip).
        """
        text = src_record.get("text", "")
        cloaked = self.augment_text(text)
        if cloaked is None or cloaked == text:
            return []

        original_source = src_record.get("metadata", {}).get("source", "unknown")
        base_meta = {
            **src_record.get("metadata", {}),
            "cloak_pair_id": int(pair_id),
            "cloak_variant": "base",
            "augmented_from": original_source,
        }
        homo_meta = {
            **src_record.get("metadata", {}),
            "source": "jionlp_homo_aug",
            "cloak_pair_id": int(pair_id),
            "cloak_variant": "homo_jionlp",
            "augmented_from": original_source,
        }
        return [
            UnifiedRecord(text=text, label=dict(src_record["label"]), metadata=base_meta),
            UnifiedRecord(text=cloaked, label=dict(src_record["label"]), metadata=homo_meta),
        ]

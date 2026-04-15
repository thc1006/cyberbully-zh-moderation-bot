# Data License Notice

This repository does **not** redistribute the raw training datasets used by
CyberPuppy. Training data processing is performed via scripts that download
from the upstream sources at your request:

- `scripts/phase2_download.py` (dry-run by default; `--live` to actually fetch)
- `scripts/phase2_build_multisource.py`
- `scripts/phase2_build_v2_2.py`

The local `data/processed/v2/*.jsonl` files produced by those scripts are
**ignored by git** (see `.gitignore`) and must not be redistributed by
CyberPuppy users — they are derivative works of upstream datasets with
heterogeneous licenses.

## Upstream data sources

| Dataset | Upstream license | Commercial use | Redistribution of raw/processed data |
|---|---|---|---|
| **COLD** ([thu-coai/COLDataset](https://github.com/thu-coai/COLDataset)) | Apache 2.0 | ✅ Allowed | ✅ Allowed with attribution |
| **ToxiCN** ([DUT-lujunyu/ToxiCN](https://github.com/DUT-lujunyu/ToxiCN)) | CC BY-NC-ND 4.0 | ❌ Prohibited | ❌ ND clause prohibits modified redistribution |
| **STATE-ToxiCN** ([shenmeyemeifashengguo/STATE-ToxiCN](https://github.com/shenmeyemeifashengguo/STATE-ToxiCN)) | CC BY-NC 4.0 (per paper); "research only" (README) | ❌ Prohibited | ⚠️ Derivatives allowed with attribution but non-commercial |
| **ToxiCloakCN** ([Social-AI-Studio/ToxiCloakCN](https://github.com/Social-AI-Studio/ToxiCloakCN)) | No formal license; derivative of ToxiCN | ❌ Inherits ToxiCN restrictions | ❌ Inherits ToxiCN ND |
| **SCCD** ([STAIR-BUPT/SCCD](https://github.com/STAIR-BUPT/SCCD)) | No formal license declared; "academic research only" per COLING 2025 paper | ❌ Default all-rights-reserved | ❌ Default all-rights-reserved |
| **CHNCI** ([zhuyiYZU/CHNCI](https://github.com/zhuyiYZU/CHNCI)) | No formal license declared | ❌ Default all-rights-reserved | ❌ Default all-rights-reserved |

## What you may / may not do

| Action | Status |
|---|---|
| Run `scripts/phase2_build_v2_2.py` locally to regenerate `data/processed/v2/*.jsonl` for your own research | ✅ |
| Train your own model on the regenerated data, for research | ✅ |
| Use the public CyberPuppy model weights (CC BY-NC-SA 4.0) for research | ✅ — see `MODEL_LICENSE` |
| Redistribute the processed JSONL files from `data/processed/v2/` in any form | ❌ Do not — this would re-publish upstream data |
| Use model weights in a commercial product | ❌ — CC BY-NC-SA 4.0 is non-commercial. Contact the CyberPuppy team for a paid commercial-variant service |
| Re-upload upstream datasets to Hugging Face or any public mirror | ❌ Do not — honor upstream authors' distribution choices |

## For upstream dataset authors

If you are an author of any of the datasets above and wish to adjust the
way CyberPuppy references, processes, or documents your dataset, please
open an issue on this repository or contact hctsai1006@cs.nctu.edu.tw.
Adjustments will be made within 7 days.

---

*Last updated: 2026-04-16*

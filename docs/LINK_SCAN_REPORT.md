# Link Scan Report - CyberPuppy Resources
**Generated**: 2025-09-24
**Status**: Deep scan completed for all links in link.md

## Executive Summary
Scanned 30+ links across datasets, models, tools, and APIs. Found several broken links and mismatched content requiring attention.

---

## 🔴 Critical Issues (Broken/404)

### Chinese Datasets
1. **COLD GitHub Repository** ❌
   - URL: `https://github.com/sunjingbo-cs/COLDataset`
   - Status: 404 Not Found
   - Impact: Primary dataset unavailable
   - Alternative: Try `https://github.com/thu-coai/COLDataset`

2. **ChnSentiCorp HuggingFace** ❌
   - URL: `https://huggingface.co/datasets/chnsenticorp`
   - Status: 404 Not Found
   - Alternative: Search HuggingFace for "chinese sentiment" or "chnsenticorp"

3. **DMSC v2 GitHub** ❌
   - URL: `https://github.com/ownthink/dmsc-v2`
   - Status: 404 Not Found
   - Alternative: Search for "Douban Movie Short Comments Dataset"

4. **NTUSD GitHub** ❌
   - URL: `https://github.com/ntusd/ntusd`
   - Status: 404 Not Found
   - Alternative: Search for "NTU Sentiment Dictionary"

---

## 🟡 Content Mismatches

### ArXiv Papers
1. **SCCD Paper (arxiv:2506.04975)** ⚠️
   - Expected: Session-level Chinese Cyberbullying Dataset
   - Actual: "Evaluating Prompt-Driven Chinese Large Language Models"
   - Note: Wrong paper linked, needs correct arxiv ID

2. **CHNCI Paper (arxiv:2506.05380)** ⚠️
   - Expected: Chinese Cyberbullying Incident Dataset
   - Actual: "EvidenceOutcomes: Clinical Trial Publications Dataset"
   - Note: Wrong paper linked, needs correct arxiv ID

3. **COLD EMNLP Paper** ⚠️
   - URL: `https://aclanthology.org/2020.findings-emnlp.363/`
   - Actual: Paper about "Temporal Reasoning in Natural Language Inference"
   - Note: Incorrect paper linked

---

## ✅ Successfully Verified Resources

### Chinese Pre-trained Models (HuggingFace)
1. **Chinese RoBERTa-wwm-ext** ✅
   - Provider: HFL (HIT-iFLYTEK)
   - Downloads: 49,327/month
   - License: Apache 2.0
   - Status: Fully accessible

2. **Chinese MacBERT-base** ✅
   - Provider: HFL (HIT-iFLYTEK)
   - Downloads: 6,505/month
   - License: Apache 2.0
   - Features: MLM as correction, WWM, N-gram masking

### NLP Tools
1. **CKIPTagger** ✅
   - Features: Word Segmentation, POS, NER
   - Performance: 97.33% F1 for segmentation
   - Installation: `pip install ckiptagger`

2. **ckip-transformers** ✅
   - Models: BERT-tiny (12M params), BERT-base (102M params)
   - Performance: Up to 97.60% word segmentation

3. **OpenCC** ✅
   - Function: Traditional/Simplified Chinese conversion
   - License: Apache 2.0
   - Multi-language support

### Interpretability Tools
1. **Captum** ✅
   - Comprehensive tutorials available
   - Covers text, vision, multimodal
   - Techniques: IG, LIME, TCAV

### Bot Development
1. **LINE Bot SDK Node.js** ✅
   - Repository: Active and maintained
   - Installation: `npm install @line/bot-sdk`
   - Requirements: Node.js 20+

### Third-party APIs
1. **Perspective API** ⚠️
   - Documentation page loaded but content unclear
   - Need to verify Chinese language support separately

2. **Kaggle Jigsaw Toxic** ⚠️
   - Page exists but content not fully loaded
   - Dataset should be accessible with Kaggle account

---

## 📋 Recommendations

### Immediate Actions
1. **Update COLD dataset link** to working alternative
2. **Find correct arxiv IDs** for SCCD and CHNCI papers
3. **Locate alternative sources** for sentiment datasets
4. **Verify Perspective API** Chinese support via official docs

### Alternative Sources to Consider
1. **For COLD**: Check THU-COAI organization on GitHub
2. **For ChnSentiCorp**: Search HuggingFace Hub directly
3. **For DMSC**: Look for mirrors on Kaggle or dataset repositories
4. **For NTUSD**: Check academic paper references

### Backup Strategy
1. Focus on accessible HuggingFace models (MacBERT, RoBERTa)
2. Use CKIP tools for Chinese NLP processing
3. Implement Captum for interpretability
4. Proceed with LINE Bot SDK for deployment

---

## 🔄 Update Status

### Links to Fix
```markdown
# Replace these in link.md:
- COLD: https://github.com/thu-coai/COLDataset (verify first)
- SCCD: Find correct arxiv paper
- CHNCI: Find correct arxiv paper
- ChnSentiCorp: Search HuggingFace for alternative
- DMSC v2: Find mirror or alternative source
- NTUSD: Search for academic sources
```

### Verified Working Links
- ✅ HuggingFace models (chinese-roberta-wwm-ext, chinese-macbert-base)
- ✅ CKIP tools (ckiptagger, ckip-transformers)
- ✅ OpenCC
- ✅ Captum tutorials
- ✅ LINE Bot SDK

---

## 🎯 Next Steps
1. Update link.md with corrected URLs
2. Create fallback dataset acquisition plan
3. Prioritize available resources for development
4. Document alternative data sources in project

**Note**: Despite some broken links, core development can proceed with:
- Available HuggingFace models
- Working NLP tools (CKIP, OpenCC)
- Accessible interpretability frameworks (Captum)
- Functional LINE Bot SDK
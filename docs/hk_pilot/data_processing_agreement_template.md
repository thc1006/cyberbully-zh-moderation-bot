# Data Processing Agreement (DPA) — Template

> **Status**: Template / starting point. **Adapt with legal counsel before signing.**
> **Aligned to**: HK Personal Data (Privacy) Ordinance (PDPO), PCPD AI Deepfake Toolkit (2025-12-17), and CyberPuppy ADR §3.5
> **Last updated**: 2026-04-16

---

## Parties

- **Data Controller** ("School" / "Customer"): _____________________
  - Address: _____________________
  - Data Protection Officer: _____________________
- **Data Processor** ("CyberPuppy" / "Provider"): hctsai1006@cs.nctu.edu.tw

## 1. Scope and purpose

The Provider operates the CyberPuppy v2.2+ Chinese cyberbullying detection
service, deployed under the chosen mode below, to assist the Controller in
identifying potential bullying / harmful content in school-related digital
communications. The Service:

- Accepts text messages submitted by the Controller's authorized systems
- Returns a multi-task classification (`toxicity`, `bullying`, `role`,
  `emotion`) and confidence scores
- Generates a SHA-256 hash of input text for correlation; **does NOT
  persist the original text in any log, database, or backup**
- Does NOT make automated disciplinary decisions; all interventions
  require human (teacher / counselor) review

**Excluded use** (not covered by this DPA):
- Automated suspension or punishment of students
- Profiling beyond the immediate moderation use-case
- Sharing predictions with third parties without case-specific consent

## 2. Deployment mode (one must be checked)

- [ ] **A. On-premise**: model + API runs on Controller's hardware. No
      personal data leaves Controller's network. Provider supplies
      Docker image, helm chart, and a quarterly update channel.
- [ ] **B. SaaS in HK**: hosted by Provider in a Hong Kong data center.
      Only SHA-256 hashes and prediction scores are stored; raw text is
      never persisted. PDPO §33 cross-border transfer not triggered.
- [ ] **C. SaaS outside HK**: requires explicit additional Schedule
      (cross-border transfer assessment) and is **not recommended** for
      pilots involving minors.

## 3. Personal data categories

| Category | Description | Retention |
|---|---|---|
| Text content (transient) | Student / staff messages submitted for analysis | Held in memory only during HTTP request; never persisted |
| SHA-256 hash | Deterministic 16-char prefix for case correlation | Up to **90 days** unless extended for active investigation |
| Prediction scores | 4-head softmax probabilities + assigned label | 90 days |
| Teacher override / appeal | Reviewer ID + override decision + timestamp | 1 year (audit trail) |
| Family consent records | Parent/guardian acknowledgment of Service usage | Term of service + 1 year |

**Special category data** (mental-health indicators inferred from "severe"
toxicity / self-harm signals): treated as PDPO sensitive-class; access
restricted to school counselor + DPO; never shared with the Provider.

## 4. Roles & responsibilities

### Controller (School) shall:
- Obtain valid family/guardian consent before submitting any minor's
  data, in compliance with PCPD's 2025 minor-data guidance
- Restrict submission of data to authorized accounts (admin, counselor,
  teacher under role-based access)
- Notify Provider within 72 hours of any suspected data breach
- Provide annual confirmation of compliance with this DPA

### Provider (CyberPuppy) shall:
- Process data **only** for the purposes in §1
- Implement the technical measures in §6
- Notify Controller within 72 hours of any incident affecting Service
  availability or data integrity
- Maintain ISO 27001-aligned operational practices (or equivalent
  documented controls)
- Permit Controller-initiated audit upon 30 days' notice (max once / year)

## 5. Data subject rights

The Controller is responsible for handling data-subject access, correction,
erasure, and objection requests under PDPO. The Provider shall:
- Respond to Controller-forwarded requests within 7 business days
- Provide self-service erasure: any hash can be purged via the
  `/v2/admin/forget` endpoint with Controller's admin credentials
- Maintain no permanent identifier that the Controller cannot purge

## 6. Technical and organizational measures

| Control | Implementation |
|---|---|
| Encryption in transit | TLS 1.3 mandatory; HSTS preload |
| Encryption at rest | AES-256-GCM on hash + score logs |
| Access control | Per-school API key, scoped to school namespace |
| Audit logging | All API calls + admin actions, retained 1 year |
| Privacy by design | SHA-256 hashing, no raw text persistence (PDPO §64 doxxing prevention) |
| Vulnerability management | Monthly dependency scan (`safety`, `pip-audit`); CVE patched within 14 days |
| Backup & restore | Daily snapshots of metadata only; 30-day retention; tested quarterly |
| Subprocessor disclosure | None for on-prem mode; HF Inference Endpoints (HK region) optional for SaaS — listed in Annex A |

## 7. Sub-processors

Provider engages no sub-processors for **on-premise** deployments.

For **SaaS** deployments, the following sub-processor list (Annex A) applies
and shall be updated with 30 days' notice to Controller:

- Hugging Face (model hosting; pull-only) — region: configurable
- Cloud-provider TBD (compute + storage) — region: HK preferred

## 8. Incident response

In the event of any of the following, Provider shall notify Controller's
DPO via email + phone within **72 hours**:
- Suspected unauthorized access to hash logs or predictions
- Service compromise affecting data integrity
- Discovery of a data leak in upstream dependencies
- Any subpoena or law-enforcement request implicating Controller's data

A post-incident report shall be delivered within 14 days, including root
cause, scope, remediation, and evidence of containment.

## 9. Term and termination

- **Term**: aligned with the Service Agreement; default 1 year, auto-renewing.
- **Termination for cause**: either party may terminate with 30 days'
  notice on material breach.
- **Upon termination**: Provider shall delete all Controller data within
  30 days and provide a deletion certificate; on-prem images may be
  retained by Controller subject to license terms in `MODEL_LICENSE`
  and `LICENSE`.

## 10. Governing law and jurisdiction

This DPA is governed by the laws of the Hong Kong Special Administrative
Region. Disputes shall be submitted to the courts of Hong Kong.

## 11. Annexes

- **Annex A** — Sub-processor list (per deployment mode)
- **Annex B** — Specific service description and SLA targets
- **Annex C** — Family consent template (Chinese + English)
- **Annex D** — Incident-response playbook (technical detail)

---

## Signatures

| | Controller | Provider |
|---|---|---|
| Name | __________________ | Tsai, Hsi-Cheng |
| Title | __________________ | Maintainer, CyberPuppy |
| Date | __________________ | __________________ |
| Signature | __________________ | __________________ |

---

*This template is provided for convenience and is not legal advice. Both
parties should engage qualified counsel before execution.*

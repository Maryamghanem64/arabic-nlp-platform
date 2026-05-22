# 🧠 Arabic NLP Comparative Platform — Project Audit (v8.3)

> **Scope:** FastAPI backend + Vue 3 frontend. Tools: **CAMeL**, **Farasa**, **Stanza**, **Qalsadi** (SinaTools is future work). Partner tool stubs exist under `backend/analyzers/`.

---

## 1) Project Overview

### 🎯 Purpose & Goals
This platform provides a **research-oriented comparative interface** for Arabic NLP outputs using a shared input text. It focuses on:
- **Morphology / lemma / root / POS / confidence** (CAMeL)
- **Segmentation and clitic splitting** (Farasa)
- **Universal POS / lemma / syntactic dependencies** (Stanza)
- **Rule-based lemma** (Qalsadi)
- Cross-tool **agreement / conflict analysis**
- **Fusion output** (heuristic scoring)
- Lightweight **evaluation against a gold dataset**

### 📚 Research Value
- Side-by-side inspection of heterogeneous tool outputs on Arabic morphology & syntax.
- Provides a practical “agreement lens” (POS/lemma/root/segmentation) to study **tool disagreement patterns**.
- Designed for extensibility: partner tools (UDPipe, AlKhalil, AraBERT, MADAMIRA) are referenced as **stubs**.

### 🧾 Tools Comparison Table

| Tool | Category | What it gives (current) | Typical strengths | Typical weaknesses |
|---|---|---|---|---|
| **CAMeL** | Statistical (MLE disambiguation) | `lemma`, `root`, `root_type`, `pos`, `gender`, `number`, `tense`, `gloss`, `confidence` | Strong morphology disambiguation; confidence scores | Can be sensitive to tokenization; limited segmentation beyond token-level |
| **Farasa** | Segmentation (SVM-like pipeline via farasapy) | `segmentation[]` per token; `segmented_text` | Strong segmentation/clitic splitting | No lemma/POS in current pipeline; segmentation alignment may drift |
| **Stanza** | Neural BiLSTM via UD pipeline | `upos`, `lemma`, `dependency{head,deprel,...}`, features mapped from `word.feats` | Contextual POS+lemma+dependency | Slower and alignment is non-trivial |
| **Qalsadi** | Rule-based (dictionary-ish) | `lemma` only (POS is currently `None` in raw output; backend normalizer may map future tags) | Lightweight & fast | Weak POS evidence (not reliable/not provided) |

---

## 2) Architecture Diagram (ASCII)

### Backend (High-level)
```text
Backend/
  main.py
    -> routes/endpoints
      -> tools functions (camel/farasa/stanza/qalsadi)
      -> fusion_system()
      -> evaluate_tools()
      -> /ui/* endpoints
    -> backend/analyzers/ (partner tools + unified adapters)
    -> backend/services/ (normalizer, alignment_engine, ui_contracts, etc.)
```

### Frontend (High-level)
```text
Frontend/
  App.vue
    -> router/index.js
      -> HomeView.vue
      -> AnalyzeView.vue
      -> CompareView.vue
    -> api/nlpApi.js (thin wrapper; mostly unused by current views)
    -> Views call FastAPI endpoints:
        /analyze-combined, /fusion, /evaluate, / (status)
```

---

## 3) Folder Structure

> ✅ = exists and correct, ⚠️ = exists but has issues, ❌ = missing, 🔧 = stub, ⏳ = future work.

### Actual (observed) vs Ideal

| Area | Actual path(s) | Status | Notes |
|---|---|---|---|
| Entrypoint | `main.py` | ⚠️ | Contains **large monolith** (tool logic + fusion + endpoints + UI endpoints) causing maintainability risk |
| Frontend root | `frontend/src/` | ✅ | Standard Vite/Vue structure |
| API base | `frontend/src/api/nlpApi.js` | ⚠️ | Functions exist but views do not consistently use it; direct axios calls are used in views |
| UI pages | `frontend/src/views/*` | ✅ | `HomeView`, `AnalyzeView`, `CompareView`, `AboutView` |
| Backend normalization | `backend/services/normalizer.py` | ✅ | Defines unified token schema for frontend |
| Agreement/alignment | `backend/services/alignment_engine.py` | ✅ | Aligns tokens and computes agreement metrics |
| UI contract helpers | `backend/services/ui_contracts.py` | ✅ | Placeholder + badge helpers |
| Tool wrappers (app/...) | `app/tools/*` | ✅ | Contains tool execution logic + normalization helpers in a parallel codepath |
| Tool execution runner | `app/services/merger_service.py` | ✅ | Runs parallel tool execution and performs fusion/evaluation for normalized/legacy compatibility |
| Gold dataset | `app/utils/constants.py` and also embedded in `main.py` | ⚠️ | Two separate sources of `GOLD_DATASET` → **inconsistent evaluation** |
| Partner tools | `backend/analyzers/*` | 🔧 | Stubs / legacy/unified adapters present; unclear if endpoints use them |
| App-specific backend configs | `backend/config/*` | ✅ | `tool_metadata.json` supports status cards |

### Ideal Structure (recommended)
- **Backend**:
  - `app/api/` (routers)
  - `app/services/` (fusion/eval/alignment)
  - `app/tools/` (tool adapters)
  - `app/schemas/` (request/response models)
  - `app/core/` (startup resource loading)
- **Single source of truth** for dataset and unified output schema.

---

## 4) Backend Audit (FastAPI)

### 🧩 main.py (central monolith)

**Purpose:** Creates FastAPI app, loads resources, defines all tool execution logic, fusion, evaluation, and endpoints including UI endpoints.

**Status:** ⚠️ Exists but has issues.

#### Issues found
1. **Monolithic file risk** 🧱
   - Tool logic, normalization calls, fusion, evaluation, and UI endpoints are all in a single file.
   - This increases the probability of regressions and makes code review difficult.

2. **Two competing normalization systems** ⚠️
   - `main.py` returns raw tool outputs sometimes (e.g., `analyze/stanza`, `analyze/qalsadi`) and normalized outputs for camel/farasa only.
   - Yet UI endpoints call `backend.services.normalizer.normalize_tool_output`.
   - Frontend views (Analyze/Compare) expect the normalized schema under `tokens[*].lemma/root/pos`.

3. **Inconsistent Qalsadi output semantics** ⚠️
   - In `main.py`, raw Qalsadi returns `pos=None` and `unvocalized`.
   - Backend normalizer expects `tok.get("freq")` for confidence; Qalsadi wrapper does not provide `freq`.

4. **Alignment mismatch vs UI compare metrics** ⚠️
   - `/ui/compare` uses alignment_engine output and agreement.
   - `CompareView.vue` uses `/analyze-combined` and `/evaluate` rather than `/ui/compare`.
   - `/evaluate` computes metrics using CAMeL vs Stanza and segmentation coverage from Farasa.
   - This can confuse users who expect token-level comparisons to match the metric definitions.

5. **`/evaluate/dataset` exists, but frontend has no UI for it** 🟡
   - Endpoint is present for supervision-grade testing, but not integrated.

6. **Potential dead code / duplication** 🧯
   - `classify_conflict` and other helpers exist both in `main.py` and in `app/utils/helpers.py`.
   - This violates DRY.

7. **`cached_analyze` cache design risk** ⚠️
   - Key includes entire `text` string, unbounded cache size.
   - No TTL or max size.

#### Fix applied / needed
- **Needed**: refactor `main.py` into routers + service modules.
- **Needed**: enforce a single consistent unified response schema for **all** analyze endpoints.
- **Needed**: unify Qalsadi confidence inputs (`freq`) or make normalizer tolerant.
- **Needed**: remove duplicated helpers or import from shared modules.

---

### 🔎 Backend tools / analyzers

The project contains **two tool stacks**:
- `main.py` tool functions: `camel_analyze`, `farasa_analyze`, `stanza_analyze`, `qalsadi_analyze`
- `app/tools/*` tool wrappers: `CamelTool`, `FarasaTool`, `StanzaTool`, `QalsadiTool`

**Status:** ⚠️
- Both likely work, but the system architecture becomes ambiguous: which stack is “canonical” for endpoints?

#### What exists vs what’s needed
- ✅ CAMeL / Farasa / Stanza implementations exist.
- ✅ Qalsadi implementation exists.
- 🔧 Partner tool stubs exist but are unclear in runtime integration.
- ⏳ SinaTools not integrated.

---

### ⚠️ Missing / questionable endpoints

| Endpoint | Expected | Observed | Status |
|---|---|---|---|
| `/evaluate/dataset` | document in UI/testing | ✅ exists | 🟢 (but not used by frontend) |
| `/analyze/camel|farasa|stanza|qalsadi` | unified schema | ⚠️ mixed normalization | 🔴 critical integration inconsistency |
| `/ui/compare` | intended research UI | ✅ exists | 🟡 not used by CompareView.vue |

---

### Response contract consistency across tools

**Finding:** UI expects token fields like:
- CAMeL: `tokens[*].analyses[0].lemma/root/pos...` in CompareView, but Normalize expects `tokens[*].lemma/root/pos`.
- AnalyzeView expects unified normalized token schema for all tools.

**Status:** 🔴 Critical
- The backend output shapes are not consistently enforced across endpoints and frontend components.

---

## 5) Frontend Audit (Vue 3)

### `frontend/src/api/nlpApi.js`
- ✅ Contains functions: `analyzeAll`, `evaluateText`, `fusionText`
- ⚠️ Views currently call axios directly and do not consistently use this wrapper.

### `HomeView.vue`
- ✅ Calls `GET /` for tool availability.
- 🟡 Shows “sinatools” as future but backend status endpoint does not list SinaTools explicitly (only implied).

### `AnalyzeView.vue`
- ✅ Uses `/analyze-combined`, then calls `/fusion`.
- ⚠️ Expects `results[tool.key].tokens[*]` to have tool-specific fields:
  - For CAMeL: expects `token.analyses[0]` (nested analyses)
  - But backend normalization schema typically flattens `lemma/root/pos` per token.

### `CompareView.vue`
- ✅ Uses `/analyze-combined` + `/evaluate`.
- ⚠️ Token alignment assumptions are index-based (camel tokens index == stanza tokens index), but backend tool pipelines may tokenize differently.

### `App.vue` + router
- ✅ Router includes expected paths.

---

## 6) API Contract

> Contract below is based on **observed endpoints in `main.py`**.

| Endpoint | Method | Params | Returns | Status |
|---|---|---|---|---|
| `/` | GET | none | `{platform, version, tools{...}, endpoints[]}` | 🟢 |
| `/analyze/camel` | GET | `text` | normalized schema via `backend.services.normalizer.normalize_tool_output` | 🟢 (but fragile) |
| `/analyze/farasa` | GET | `text` | normalized schema | 🟢 |
| `/analyze/stanza` | GET | `text` | **raw** output (`status ok`, `tokens[]` with `upos/lemma/dependency`) | 🟡 mismatch risk |
| `/analyze/qalsadi` | GET | `text` | raw output (`tokens[]` with `lemma`, `pos=None`, `unvocalized`) | 🟡 mismatch risk |
| `/analyze/{tool}` | GET | `tool,text` | depends on tool; mostly raw | 🟡 |
| `/analyze-combined` | GET | `text` | `{camel,farasa,stanza,qalsadi}` raw tool outputs | 🟡 |
| `/compare` | GET | `text,tools` | raw tool outputs in `results` | 🟡 |
| `/fusion` | GET | `text` | `{input, qalsadi, fusion_result{fusion:[...]}}` | ⚠️ likely shape drift |
| `/evaluate` | GET | `text` | `{input,evaluation:{pos_agreement_pct,...}}` | 🟢 |
| `/evaluate/dataset` | GET | none | dataset-level averaged metrics | 🟢 |
| `/export` | GET | `text,format=json|csv` | downloadable analysis | 🟡 output depends on fusion/token shape |
| `/cache/clear` | POST | none | `{status}` | 🟢 |

---

## 7) Tool Output Shapes (Observed / Contracted)

### CAMeL
Observed raw token shape in `main.py`:
- `tokens[]` where each token is:
  - `surface`
  - `analyses[]` (top 3)
    - `{pos, lemma, root, root_type, gloss, gender, number, tense, confidence, confidence_level, corrections}`
  - `segmentation` (currently `[token]`)

### Farasa
Observed raw token shape in `main.py`:
- `tokens[]` entries:
  - `surface`
  - `segmentation[]` (clitic split)

### Stanza
Observed raw token shape in `main.py`:
- `tokens[]` entries:
  - `surface`, `lemma`, `upos`, `xpos`,
  - morphological features: `gender, number, tense, person, voice, case, definite, aspect`
  - `dependency{head, head_text, deprel}`

### Qalsadi
Observed raw token shape in `main.py`:
- `tokens[]` entries:
  - `surface`, `lemma`, `pos=None`, `stem=None`, `unvocalized` (duplicated surface)

---

## 8) Issues Log

### 🔴 Critical (fixed)
- ✅ None verified as fixed (audit only). 

### 🟡 Important (pending)
1. **🔴 Unified schema inconsistency across endpoints**
   - Some endpoints return raw output, others return normalized output.
   - Frontend expects unified shapes; it likely breaks in edge cases.

2. **Token alignment by index instead of alignment engine**
   - CompareView assumes CAMeL/Stanza/Farasa token arrays align by index.
   - This is unsafe for Arabic tokenization differences.

3. **Qalsadi confidence mapping mismatch**
   - Normalizer references `tok.get("freq")` but raw Qalsadi provides neither `freq` nor `confidence` fields.

### 🟢 Minor (nice to have)
- Cache lacks max-size/TTL.
- Partner tools are stubbed without clear UI integration.
- Duplicate helper logic across `main.py` and `app/*`.

---

## 9) Integration Status

### 🧩 My Tools (CAMeL)

| Tool | Status | Endpoint | Output |
|---|---|---|---|
| CAMeL | ✅ loaded (at startup) | `/analyze/camel`, `/analyze-combined` | raw with `analyses[]` + normalized option inside endpoint |
| Farasa | ✅ loaded (at startup) | `/analyze/farasa`, `/analyze-combined` | `segmentation[]` |
| Stanza | ✅ loaded (at startup) | `/analyze/stanza`, `/analyze-combined` | UD-style `upos/lemma/dependency` |
| Qalsadi | ✅ loaded (at startup) | `/analyze/qalsadi`, `/analyze-combined` | lemma-only tokens |

### 🔧 Partner Tools

| Tool | Status | Owner | Integration |
|---|---|---|---|
| UDPipe | 🔧 stub | backend/analyzers | not confirmed via runtime endpoints |
| AlKhalil | 🔧 stub | backend/analyzers | not confirmed via runtime endpoints |
| AraBERT | 🔧 stub | backend/analyzers | not confirmed via runtime endpoints |
| MADAMIRA | 🔧 stub | backend/analyzers | not confirmed via runtime endpoints |

---

## 10) Merge Plan

### ✅ Pre-merge checklist
- [ ] Run backend smoke tests (`curl` endpoints).
- [ ] Run frontend dev build (`npm run build`) to ensure no runtime schema mismatch.
- [ ] Validate token schema contracts for:
  - `/analyze-combined`
  - `/fusion`
  - `/evaluate`

### 🧪 Merge day steps
1. Tag release `v8.3-audit`
2. Implement schema alignment refactor (recommended)
3. Update frontend to rely on unified normalized tokens only.
4. Run compare/evaluate flows end-to-end.

### ✅ Post-merge verification
- Confirm:
  - Compare view table renders correctly
  - Metrics cards match evaluation output
  - Export returns valid JSON/CSV

### 🌿 Git workflow
- Branch: `blackboxai/audit-schema-refactor`
- PR includes:
  - schema contracts
  - adapter updates
  - frontend mapping fixes

---

## 11) Defense Readiness

Feature readiness rating (0–100%):

| Feature | % Done | Notes |
|---|---:|---|
| Tool execution stability | 85 | All core tools load; some runtime failures possible if models missing |
| Unified response schema | 55 | Inconsistent normalized/raw responses |
| Agreement/fusion research value | 75 | Fusion and agreement logic exist, but mapping may be inconsistent |
| Evaluation dataset endpoint | 80 | Endpoint exists; frontend not using it directly for token-level comparison |
| Frontend UX | 78 | UI is polished but depends on schema assumptions |
| Extensibility (partner tools) | 30 | Stubs not integrated with runtime endpoints/UI |

---

## 12) Recommendations

### Priority 1 (must before defense)
1. **Enforce one canonical unified token schema** ✅
   - Ensure `/analyze-combined`, `/compare`, `/analyze/{tool}` return consistent `tokens[*]` with the same shape.

2. **Fix token alignment logic** ✅
   - Use `backend/services/alignment_engine.py` outputs for CompareView, not index-based matching.

3. **Repair Qalsadi confidence/POS contract** ✅
   - Either:
     - provide `freq`/confidence in raw output, or
     - update normalizer to avoid referencing missing fields.

### Priority 2 (should)
- Remove duplicated code paths: choose either `main.py` tool functions or `app/tools/*` wrappers.
- Add explicit frontend schema versioning.

### Priority 3 (nice to have)
- Add `/ui/*` integration to CompareView and allow metrics row-click drill-down.
- Add cache max-size/TTL.
- Add partner tool roadmap UI.

---

## 13) Test Commands

> Assumes FastAPI runs on `http://127.0.0.1:8000`.

```bash
# Backend smoke
curl "http://127.0.0.1:8000/"

# Tool analyses
curl "http://127.0.0.1:8000/analyze/camel?text=كتب"
curl "http://127.0.0.1:8000/analyze/farasa?text=كتب"
curl "http://127.0.0.1:8000/analyze/stanza?text=كتب"
curl "http://127.0.0.1:8000/analyze/qalsadi?text=كتب"

# Combined analysis
curl "http://127.0.0.1:8000/analyze-combined?text=ذهب%20محمد%20إلى%20المدرسة"

# Fusion
curl "http://127.0.0.1:8000/fusion?text=ذهب%20محمد%20إلى%20المدرسة"

# Evaluation
curl "http://127.0.0.1:8000/evaluate?text=ذهب%20محمد%20إلى%20المدرسة"
curl "http://127.0.0.1:8000/evaluate/dataset"

# Export
curl "http://127.0.0.1:8000/export?text=كتب&format=json" -I
curl "http://127.0.0.1:8000/export?text=كتب&format=csv" -I

# Cache
curl -X POST "http://127.0.0.1:8000/cache/clear"
```

---

## 14) Known Limitations

- 🐢 **Stanza is slow on first request** (model warm-up).
- 🧾 **Qalsadi lemma-only** (POS not reliably available; stems/confidence may be missing).
- ⏳ **SinaTools excluded** due to ~880MB model size and Windows loading constraints.
- 🔧 **Partner tools pending** (UDPipe, AlKhalil, AraBERT, MADAMIRA stubs not confirmed end-to-end).

---

## ✅ Summary for Supervisor
- The platform is **research-functional** with working CAMeL/Farasa/Stanza/Qalsadi pipelines and comparative/fusion/evaluation endpoints.
- Main technical risk for defense is **schema & alignment inconsistency**: the backend does not consistently guarantee a unified token contract, and the frontend Compare view uses **index-based alignment** instead of robust alignment.
- Addressing unified schema + alignment will significantly improve scientific credibility and reproducibility.


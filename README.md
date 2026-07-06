# Arabic NLP Comparative Analysis Platform

This platform compares outputs of Arabic NLP analyzers and performs capability-aware expert fusion. It does not claim ground-truth correctness.

The project is a FastAPI + Vue 3 research workbench for inspecting Arabic NLP evidence from multiple analyzers, comparing agreement and conflict patterns, producing an auditable expert-fusion decision, and reporting capability-aware agreement metrics.

## Research Scope

The platform is designed for comparative analysis, not gold-standard evaluation. Its outputs should be interpreted as analyzer evidence:

- Raw and normalized outputs from Arabic NLP tools
- Token-level agreement and disagreement across comparable tools
- Feature-specific expert fusion for segmentation, lemma, root, POS, morphology, and dependency evidence
- Capability-aware evaluation that excludes unsupported, lazy, unavailable, or excluded tools from scoring
- Frontend visualization of tool outputs, conflicts, selected sources, confidence, and methodology notes

## Tool Status

| Tool | Status | Main capability | Used in Fusion? | Used in Evaluation? | Notes |
| --- | --- | --- | --- | --- | --- |
| CAMeL | Working | Morphology, lemma, root, POS, gloss | Yes | Yes, for supported morphology, lemma, root, and POS evidence | Primary lexical and morphological evidence source. |
| Farasa | Working, segmentation-focused, may be slow | Clitic segmentation | Yes, mainly segmentation | Yes, for segmentation coverage when available | Java-backed tool. Prewarm before demo if possible; timeout or degraded runs are excluded from scoring rather than counted as wrong. |
| Stanza | Working | POS, lemma, Universal Dependencies syntax | Yes | Yes, for POS, lemma, and dependency-supported metrics | Tokenization and multi-word-token behavior can differ from other analyzers. |
| Qalsadi | Working partial | Lemma and rule-based lexical support | Yes, mainly lemma fallback/support | Yes, for lemma where comparable values exist | Does not provide the full morphology/root/dependency profile expected from larger analyzers. |
| AlKhalil | Working | Rule-based Arabic morphology, lemma, root, POS evidence | Yes | Yes, for morphology, lemma, root, and POS where normalized | Useful root and morphology evidence; POS labels require canonical normalization before conflict reporting. |
| UDPipe | Working | POS and dependency syntax | Yes | Yes, for POS, lemma, and dependency-supported metrics | UD tokenization and dependency conventions can differ from Stanza. |
| SinaTools | Working lazy-loaded local resource | Lemma, root, POS lexical evidence | Yes when loaded | Yes when loaded and capability-supported | Heavy local resource. It reports `lazy_not_loaded` or `loading` until explicitly preloaded; lexical POS can disagree with contextual/UD tools. |
| AraBERT | Working contextual support only | Contextual transformer evidence | No direct morphology fusion; contextual support only | Excluded from morphology/POS/lemma/root metrics | Base AraBERT does not provide lemma, root, POS, segmentation, or dependency without a fine-tuned task head. |
| MADAMIRA | Excluded, missing licensed resources | Morphological analysis if licensed resources are present | No in the current defense configuration | No | The wrapper checks for licensed resources, but the project should document it as excluded unless those resources are legally installed. |

## Architecture Overview

```text
Arabic input
  -> analyzer adapters
  -> normalization and token alignment
  -> comparison of agreement/conflicts
  -> expert fusion
  -> capability-aware evaluation
  -> frontend visualization and export
```

The backend keeps each analyzer behind an adapter boundary. Analyzer outputs are normalized into a shared token schema before comparison, fusion, or evaluation. The frontend then presents the research flow as:

```text
Input Arabic Text -> Tool Outputs -> Compare Agreement/Conflicts -> Expert Fusion Decision -> Capability-Aware Evaluation
```

## Expert Fusion

Expert Fusion is implemented as feature-specific, capability-weighted fusion. It is better than simple priority fusion because it does not assume one analyzer is best for every linguistic feature.

The fusion layer uses different expert strategies for:

- Segmentation: Farasa is the strongest segmentation anchor, with morphology tools used as support when present.
- Lemma: CAMeL, SinaTools, Qalsadi, Stanza, UDPipe, and AlKhalil can contribute according to capability weight.
- Root: CAMeL, SinaTools, and AlKhalil are prioritized; root evidence for functional words is deemphasized.
- POS: Stanza, UDPipe, CAMeL, SinaTools, and AlKhalil are compared after normalization.
- Dependency: UDPipe and Stanza are treated as syntax specialists.
- Morphology: CAMeL, AlKhalil, and SinaTools provide the main morphology evidence.

Each fused token preserves the final selected value, selected source, supporting tools, disagreeing tools, candidate evidence, strategy, confidence score/level, and decision trace. Confidence is an evidence-summary signal, not a correctness label.

## Capability-Aware Evaluation

Evaluation is agreement-based, not gold-standard accuracy. Metrics are computed only over analyzers that support the evaluated feature and produced comparable evidence.

Unsupported, lazy, excluded, unavailable, timeout, or degraded tools are not counted as wrong outputs. This is important for:

- AraBERT, which is contextual support only and should not enter morphology, lemma, root, POS, segmentation, or dependency metrics.
- MADAMIRA, which is excluded because licensed resources are missing.
- SinaTools, which contributes only after its local resource is loaded.
- Farasa, which contributes mainly to segmentation and is excluded from scoring if timeout/degraded.

Reported metrics include POS agreement, lemma match, root agreement, segmentation coverage, active tools, excluded tools, capability contributors, metric contributors, and methodology notes.

## Frontend Research Flow

The Vue frontend is organized around the defense narrative:

- Dashboard/Home: tool health, capability overview, and research workflow.
- Analyze: individual or combined analyzer output without fabricated missing fields.
- Compare: aligned analyzer evidence and real feature-level conflicts.
- Smart Analysis / Smart Fusion: selected fusion values, sources, supporting/disagreeing tools, confidence, and decision trace.
- Evaluate: capability-aware metrics, excluded tools, contributors, and methodology note.
- About/Reports: project framing and reproducibility explanation.

Recommended UI wording discipline:

- "No data available" should mean unsupported or unavailable evidence, not analyzer failure by default.
- AraBERT null morphology should be shown as "Contextual support only."
- MADAMIRA should be shown as "Excluded: missing licensed resources."
- SinaTools should be shown as "Lazy-loaded local resource" until preloaded.

## Demo Notes

### Backend

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\python.exe install_models.py
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

For a defense demo:

- Start the backend before the frontend.
- Prewarm Java-backed tools, especially Farasa, if using them live.
- Preload SinaTools only if the local lemma resource is needed for the demonstration.
- Present MADAMIRA as excluded unless licensed resources are installed.
- Present AraBERT as contextual transformer support only.

## Known Limitations

- The platform does not evaluate against a human-annotated gold standard.
- Agreement does not prove linguistic correctness.
- AraBERT does not produce lemma, root, POS, segmentation, or dependency without a fine-tuned head.
- MADAMIRA remains excluded in the current configuration because required licensed resources are missing.
- SinaTools is a local lexical resource and may disagree with contextual or UD-oriented POS analyzers.
- Farasa can be slower than lightweight Python analyzers and should be prewarmed for demo stability.
- Stanza and UDPipe can differ because of tokenization, multi-word-token handling, and UD conventions.
- Segmentation disagreement often reflects clitic segmentation style rather than analyzer failure.
- The repository still contains both `app/` and `backend/` namespaces; this is acceptable for the current defense but should be consolidated in future maintenance.

## Repository Layout

- `app/`: FastAPI routes, application services, startup, and runtime analyzer facades
- `backend/`: analyzer wrappers, normalization, alignment, comparison, evaluation support, schemas, and configuration
- `frontend/`: Vue 3 user interface
- `docs/`: architecture and methodology documentation
- `scripts/`: demo/prewarm helpers
- `requirements.txt`: backend Python dependencies
- `optional_requirements.txt`: optional research integrations

## Documentation

- [Architecture audit](./docs/architecture_audit.md)
- [Evaluation methodology](./docs/evaluation_methodology.md)
- [Installation guide](./INSTALLATION_GUIDE.md)

## License

See [`LICENSE`](./LICENSE). The repository currently includes a license notice rather than a published open-source license.

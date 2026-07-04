# Arabic NLP Platform

Arabic NLP Platform is a FastAPI + Vue 3 application for comparing the outputs of multiple Arabic NLP analyzers.

## What It Includes

- FastAPI backend with analyzer adapters, alignment, comparison, evaluation, and fusion endpoints
- Vue 3 frontend for analysis, comparison, evaluation, and project reporting
- Unified API response envelopes
- Token-level comparison and evaluation summaries
- Export support for JSON and CSV

## Repository Layout

- `app/` backend application layer and API routes
- `backend/` normalization, alignment, comparison, and evaluation services
- `frontend/` Vue 3 user interface
- `docs/` architecture and methodology documentation
- `requirements.txt` backend Python dependencies
- `optional_requirements.txt` optional partner and research integrations
- `setup.ps1` Windows setup helper

## Requirements

- Python 3.10+ recommended
- Node.js 20+ recommended
- Java if you plan to use Java-backed tools such as AlKhalil or MADAMIRA

## Analyzer Status

| Tool | Status | Notes |
| --- | --- | --- |
| CAMeL | ✅ Working | Loads and returns token-level morphology in the backend. |
| Farasa | ✅ Working | Uses the local Farasa JAR directly with a backend timeout; isolated and combined smoke tests returned successfully. |
| Stanza | ⚠️ Partial | Arabic model is detected, but cold analysis hung under current low-memory conditions; `/analyze-combined` skips it below the `STANZA_LOW_MEMORY_THRESHOLD_MB` safety floor. |
| Qalsadi | ✅ Working | Existing analyzer remains registered and available. |
| AlKhalil | ⚠️ Partial | Java bridge/resources detected; remains optional and should be verified before demo use. |
| UDPipe | ✅ Working | Local `arabic.udpipe` model loads successfully. |
| MADAMIRA | ⚠️ Partial | Working through the bundled Java persistent server with `-msaonly`; MSA-only mode limits dialect/resource coverage. |
| SinaTools | ❌ Blocked | Morphology files are present, but the official morphology API still times out after 60s; NER bundle `Wj27012000.tar` is also missing. |
| AraBERT | ✅ Working | Cached `aubmindlab/bert-base-arabertv2` fill-mask model works as contextual support; lemma/POS/root intentionally remain null. |

## Quick Start

1. Create and activate a Python virtual environment.
2. Install backend dependencies from `requirements.txt`.
3. Run `install_models.py` if your local environment needs downloadable analyzer models.
4. Install frontend dependencies inside `frontend/`.
5. Start the FastAPI backend.
6. Start the Vue frontend.

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

## Documentation

- [Architecture audit](/docs/architecture_audit.md)
- [Evaluation methodology](/docs/evaluation_methodology.md)
- [Installation guide](./INSTALLATION_GUIDE.md)

## License

See [`LICENSE`](./LICENSE). The repository currently includes a license notice rather than a published open-source license.

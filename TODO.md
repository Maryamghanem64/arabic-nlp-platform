# Refactor plan: main.py monolith -> app/api + app/services + app/core

- [ ] Step 1: Create `app/core/startup.py` to load NLP resources and expose globals used by moved logic.
- [ ] Step 2: Create `app/services/fusion_service.py` with fusion logic (scoring, confidence, fuse_token, fusion_system).
- [ ] Step 3: Create `app/services/eval_service.py` with evaluation logic (compute_prf, evaluate_tools, GOLD_DATASET, evaluate_dataset).
- [ ] Step 4: Create `app/api/analyze.py` with APIRouter endpoints for `/analyze/*` and `/analyze-combined`.
- [ ] Step 5: Create `app/api/fusion.py` with endpoint `/fusion`.
- [ ] Step 6: Create `app/api/evaluate.py` with endpoints `/evaluate` and `/evaluate/dataset`.
- [ ] Step 7: Create `app/api/ui.py` with endpoints `/ui/*` (ui analyze/compare/fusion) and UI helpers.
- [ ] Step 8: Ensure a shared cache + base tool functions exist in services/core without changing behavior.
- [ ] Step 9: Replace `main.py` with ~20 lines wiring routers only.
- [ ] Step 10: Run a quick import check (e.g. `python -c "import main; ..."`) and ensure server starts.
- [ ] Step 11: Smoke test key endpoints return expected keys (shape preservation).


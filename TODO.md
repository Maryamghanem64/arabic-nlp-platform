# TODO

- [ ] Implement canonical AlKhalil POS extraction helpers in `backend/services/normalizer.py`:
  - [ ] `normalize_alkhalil_pos(value)` to map to UD labels / return None for invalid/long descriptions
  - [ ] `extract_alkhalil_canonical_pos(token, context_pos_votes=None)` scanning all analyses and selecting via context votes
- [ ] Update `app/services/eval_service.py` POS extraction to use canonical AlKhalil POS selection (with context votes) so conflicts use UD POS only.
- [ ] Update `app/services/fusion_service.py` conflict normalization/cleaning to rely on canonical AlKhalil POS mapping (no raw Arabic POS in comparisons).
- [ ] Run py_compile for the 3 modified files.
- [ ] Runtime verification for `/fusion` and `/evaluate` using the provided Arabic test string.


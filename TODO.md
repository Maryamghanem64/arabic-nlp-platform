# TODO

## Frontend audit: Fix analyzer rendering (Farasa segmentation)

- [ ] Identify and fix root cause for Farasa segmentation showing `—` instead of `ال + طالب`.
- [ ] Implement defensive segmentation extraction in `frontend/src/views/AnalyzeView.vue`.
- [ ] Ensure placeholder `—` only appears when segmentation truly missing.
- [x] Validate on sample: `قرأ الطالب الكتب في المكتبة` (Farasa segmentation should show joined tokens).
- [ ] Continue audit for other analyzers/cards if any other fields still render as `—`.


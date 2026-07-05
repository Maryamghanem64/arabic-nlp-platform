- [ ] Inspect eval_service.py current logic for active/excluded/tools, metric_contributors, and conflict fields.
- [ ] Refactor conflict generation to store normalized values as value_a/value_b and raw Arabic descriptions only as raw_value_a/raw_value_b.
- [ ] Ensure status "excluded" never counts as active (filter it out from active_tools and scoring contributors).
- [ ] Force madamira to be excluded regardless of status.
- [ ] Make AraBERT contextual only (exclude from metric contributors and feature contributors).
- [ ] Normalize AlKhalil POS before creating conflicts (ensure normalized POS used in scoring/conflicts; raw stored separately).
- [ ] Update return payload fields accordingly and keep metric_contributors consistent with requirements.
- [x] Run a quick syntax check for the updated file.



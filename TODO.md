# Stanza Improvement TODO

## Plan Implementation Steps

- [x] Step 1: Add new Pydantic models (StanzaToken, Dependency, StanzaResponse) to main.py
- [x] Step 2: Add parse_feats helper function to main.py
- [x] Step 3: Replace stanza_analyze with new implementation using StanzaToken/StanzaResponse
- [x] Step 4: Update /analyze-stanza endpoint response_model to StanzaResponse
- [x] Step 5: Ensure /compare uses new stanza_analyze (returns new format as requested)
- [ ] Step 6: Test endpoints (run server, curl /analyze-stanza?text=كتب, verify output matches good example)
- [ ] Step 7: Complete - attempt_completion

**Notes**: Keep CAMeL/Farasa unchanged. Python 3.10/Pydantic v2. No new deps.

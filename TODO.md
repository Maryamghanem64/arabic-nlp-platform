# TODO - SinaTools Fix & Preload UX

## Task A — Fix SinaTools properly
- [ ] Update `app/tools/sinatools_tool.py` (re-export only)
  - [x] Implement local pickle priority ordering + required status fields in `backend/analyzers/sinatools_tool.py`
  - [x] Ensure background preload updates those fields


## Task B — Add backend endpoints
- [ ] Add `POST /tools/sinatools/preload`
- [ ] Add `GET /tools/status`
- [ ] Ensure `/tools/status` returns statuses for all tools, especially SinaTools: `lazy_not_loaded`, `loading`, `loaded`, `error`, `excluded`

## Task C — Fix `/analyze-combined`
- [ ] Update `app/api/analyze.py`
  - [ ] Never call SinaTools synchronously if not loaded
  - [ ] If SinaTools not loaded: return `status="lazy_not_loaded"` + reason
  - [ ] If SinaTools loading: return `status="loading"` + reason
  - [ ] If SinaTools loaded: run it normally
  - [ ] MADAMIRA: return excluded immediately and never run
  - [ ] Core tools run normally and return real output

## Task D — Add preload script
- [ ] Create `scripts/preload_sinatools.py`
  - [ ] Load pickle once
  - [ ] Print model_path + load time + dictionary size
  - [ ] Print memory usage if available
  - [ ] Run sample: `وجدت المعلمة طالبة مجتهدة في الفصل`

## Task E — Frontend Analyze UI
- [ ] Update `frontend/src/api/nlpApi.js` with `POST /tools/sinatools/preload`
- [ ] Update `frontend/src/composables/useToolStatus.js` to poll `GET /tools/status`
- [ ] Update `frontend/src/views/AnalyzeView.vue`
  - [ ] If SinaTools `lazy_not_loaded`: show card + button to load
  - [ ] While loading: show spinner text
  - [ ] Poll status until loaded; show loaded message and prompt rerun

## Verify
- [ ] Restart backend + frontend
- [ ] `GET /tools/status`
- [ ] `GET /analyze-combined?text=وجدت%20المعلمة%20طالبة%20مجتهدة%20في%20الفصل`
- [ ] Ensure page does not hang
- [ ] Click frontend preload and verify status transitions and rerun result includes SinaTools


# TODO - Arabic NLP Platform Dependency + Integration Audit

## Step 1: AlKhalil centralized jar resolution
- [x] Refactor `app/tools/alkhalil_tool.py` to use `backend/config/tool_paths.py` (AlKhalilPaths) and remove duplicated casing/hardcoded candidate list.
- [x] Ensure runtime AlKhalil and startup diagnostics both use the same resolver.


## Step 2: UDPipe centralized model resolution + lazy loading
- [x] Refactor `app/tools/udpipe_tool.py` to use `backend/config/tool_paths.py` (UDPipePaths) and remove hardcoded candidates.
- [x] Remove duplicated env-setting logic or align both adapters to centralized resolver.

- [ ] Add startup diagnostics + graceful missing-model messages.

## Step 3: CAMeL compatibility shim
- [ ] Implement defensive imports for CAMeL API changes (AR_DIAC_CHARSET removed/renamed).
- [ ] Add version detection and backwards-compatible imports.

## Step 4: Farasa + emoji compatibility shim
- [ ] Patch Farasa integration to avoid importing missing `emoji.EMOJI_DATA`.
- [ ] Add dependency pin (requirements update) for compatible emoji version OR runtime fallback.

## Step 5: Architecture cleanup
- [ ] Centralize all tool configuration (jar/model paths) to `backend/config/tool_paths.py`.
- [ ] Ensure heavy tools are lazy-loaded.
- [ ] Improve logging and startup health checks.

## Step 6: Verification
- [ ] Run `python startup_check.py` (or `python main.py` startup) and confirm:
  - AlKhalil jar exists -> status ok
  - UDPipe model path properly resolved
  - CAMeL and Farasa import errors no longer crash startup
- [ ] Run minimal analyzer calls for each tool to validate outputs.


# TODO

## Bug 1 — Farasa loads twice (double execution)
- [ ] Root-cause pinpoint in call chain for `GET /evaluate`.
- [ ] Implement race-condition safe per-text request deduplication at the endpoint call-site.
- [ ] Ensure each tool run occurs once per request text even under concurrent requests.
- [ ] Update logs/behaviour expectation: each tool load/run once.

## Bug 2 — excluded_tools always empty
- [ ] Fix `evaluate_tools()` to consider statuses of *all* optional tools (alkhalil, udpipe, arabert, madamira, sinatools).
- [ ] Update function signature to accept all tool results (option with minimal caller changes).
- [ ] Update all call sites accordingly.
- [ ] Ensure `excluded_tools` lists every tool whose status is error/unavailable/future_work/lazy.


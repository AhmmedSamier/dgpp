# Completed benchmark matrix

All 29 deployments and 336 measurement reports passed validation. No benchmark groups remain pending. The [overview](../../../docs/benchmarks.md) contains the completed serving, decode-mode, context-scaling and cold-prefill tables.

| Check | Final result | Evidence |
|---|---|---|
| Serving concurrency | C1, C2, C4 and C8 for every deployment | [Coverage and per-deployment results](README.md) |
| Report validation | 336 passed; 0 pending | [Measurement audit](diagnostics/measurement-audit.json) |
| Documentation | 27 tables; 65 local links; 45 historical quality rows preserved | [Document audit](diagnostics/document-audit.json) |
| Prefill settings | Resolved busy/idle budgets recorded for all 29 deployments | [Startup records](diagnostics/resolved-prefill-budgets.json) |
| Measurement memory | Every published launch passed its process-swap check; excluded swapping attempts contribute no results | [Source records](result-provenance.json) |
| Model cleanup | All three session-downloaded models and their generated caches removed; original snapshots and resident-file sizes preserved | [Storage audit](diagnostics/storage-preservation-audit.json) |
| Original service | Original four-node GLM Flash configuration restored with the validated fixed binary; inference and all ranks checked | [Restoration receipt](diagnostics/original-service-restoration/restored.json) |
| Service persistence | All ranks survived 45 seconds after monitoring SSH sessions closed; process and cgroup swap remained zero | [Post-logout process checks](diagnostics/original-service-restoration/attempt-2/processes-after-logout.json) |

The first restoration attempt ended when rank 2's user manager stopped after SSH logout. Its journal records the manager killing the serving scope, with zero memory swap in that scope. Lingering is now enabled for the serving account on all four nodes; OS swap remains enabled. [Journal](diagnostics/original-service-restoration/failed-attempt-rank2-journal.log), [setting changes](diagnostics/original-service-restoration/user-service-persistence.json).

The route-trace and prefill fixes are pushed through `07a32fb`. The documentation, harness and validation records accompany this completed matrix. Full GLM's remaining long-context performance investigation is tracked in [issue #96](https://github.com/HawkBearPig/dgpp/issues/96).

[Completion record and evidence hashes](completion.json).

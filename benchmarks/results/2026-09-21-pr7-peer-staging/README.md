# Peer staging cleanup across login shells

PR #7 identified that an unmatched `serve_rank*.ops` shell glob fails on
zsh peers. Quoting the entire operand to `rm` avoids that error but leaves
existing operation streams behind, allowing a failed startup to reuse old
files as evidence during `down`.

The follow-up passes the quoted pattern to `find`, limits deletion to regular
files directly in the staging directory, and succeeds when no files match.
The launcher tests execute the actual peer staging command in Bash, Bash with
`failglob`, and zsh when available. They cover missing, empty and populated
directories, remove two stale streams, and preserve unrelated and nested files.

Validation on 2026-09-21:

- Before the fix, the new Bash and `failglob` tests failed because stale
  operation streams survived staging.
- After the fix, all 19 launcher tests and 25 site configuration tests passed.
  The zsh cases ran with zsh 5.9 and its default `nomatch` option enabled.
- The combined merge with master `e91ebb5` passed all 63 tests from
  `PYTHONPATH=tests/python python3 -m unittest launcher_test site_env_test portability_test`.
  The PR's older base lacked two script-index links; master already fixes that
  unrelated portability-test failure.
- Python compilation of the launcher and its test, and `git diff --check`, passed.

The checks use local shells and stub file transfers; no serving deployment
or GPU/RDMA workload was started.

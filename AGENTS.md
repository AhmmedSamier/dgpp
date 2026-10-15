# Working in this repository

Use the primary repository checkout for development. Do not create Git
worktrees or use linked checkouts to isolate tasks. Preserve existing work
when updating branches, and leave completed changes committed and pushed
when the user requests that outcome.

Keep generated benchmark results, profiles and run logs untracked. Do not
force-add ignored benchmark artifacts. Summarize validation in the PR and
keep reusable documentation in `docs/`, without links to local run artifacts.

# Campaign interruptions

## GLM-4.7 node outage

At 2026-09-22 03:27:07 UTC, the first approximately 8K-token cold-prefill
request failed with `engine_failure`. The transport reported an RDMA retry
limit exceeded on the connection to rank 2 (`192.168.88.13`, `spark-a289`).
That node was also unreachable over SSH and ICMP. The healthy ranks exited,
and the campaign stopped before the next deployment.

The user reported that the machine was hot to the touch and restarted it.
It was reachable at 03:32:52 UTC, with a GPU temperature of 50°C and no active
GPU thermal-slowdown flags. The previous boot's journal contained no kernel
entries after 03:00 UTC and ended with routine system activity at 03:25:01.
The available logs do not establish the cause of the outage.

The interrupted deployment was confirmed stopped at 03:33:17 UTC. Its three
recoverable operation streams match. Rank 2's stream was lost during the
restart, so that attempt cannot pass the four-rank identity check.

The incomplete attempt is retained in
[attempt-20260922T0327](raw/glm47-w4/performance/attempt-20260922T0327/),
including the server logs, command results, previous-boot journal excerpt
and recovery shutdown log. A fresh GLM-4.7 sweep was attempted after the
pause described below.

[GPU telemetry snapshots](raw/thermal-snapshots.json) taken after the outage
describe the healthy nodes at that instant. Their cumulative counters do
not provide a before/after comparison for the campaign.

## Recurrence during the resumed GLM-4.7 sweep

After the user-requested pause, the campaign resumed at 05:15 UTC with the
same source revision and binary. All four nodes were idle, with GPU
temperatures of 42, 39, 39 and 48°C and no active thermal-slowdown flags.
See the [pause and resume record](raw/user-pause.json).

The fresh GLM-4.7 run completed all 45 greedy phases and all three 2K
cold-prefill samples. At 05:24:38 UTC, its first 8K cold-prefill request
failed with the same RDMA retry-limit error to rank 2, lane 1. Both
192.168.88.13 and 192.168.89.13 then failed SSH connection attempts; the
primary address also failed ICMP. The last GPU sample before the failure
was 75°C, with no active thermal-slowdown flags. This is a repeatable
workload boundary, but the available evidence does not establish the cause.

Ranks 0, 1 and 3 were confirmed stopped. Their recovered operation streams
match. The second attempt is preserved in
[attempt-20260922T0524](raw/glm47-w4/performance/attempt-20260922T0524/).

Node 3 rebooted at 05:57:31 UTC and was reachable at 05:58:39, reporting
39°C, approximately 117.8 GiB of available RAM and no swap usage. The old
deployment was confirmed stopped at 06:01:03. Rank 2's operation stream was
lost on reboot. Recovered system-activity samples show about 30–31 GiB of
available RAM several minutes before both outages, without swap traffic or
page-reclaim scans. No OOM, GPU Xid or kernel-panic signature was found in
the saved journal. These observations do not establish the cause; memory
at the failing prefill was not captured. See the
[environment comparison](environment.md) for the evidence and its limits.

GLM-4.7 is excluded from further performance and quality measurements at
the user's request. Other models remain in the matrix. Subsequent
four-node launches record host memory and pressure approximately every
second, GPU state approximately every five seconds, and Node 3's live
kernel journal to help investigate any recurrence.

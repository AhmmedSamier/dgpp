# Qwen3.8-27B drafter: the one-node profile, the sampled proposals and the GEMV staging (2026-10-05)

Raw legs behind the 2026-10-05 changes (CHANGELOG "sampled proposals, the one-node graph
template, the first-miss histogram"). Server directories, nsys reports and sqlite exports are
not committed; every other log is.

| leg | what | result |
|---|---|---|
| `mtbench/`, `mtbench.log` | MT-Bench turn-1 (writing / roleplay / humanities / stem, max_tokens 384) on the one-node drafter, graph engine, argmax walk | think-sampled 2.80 tok/pass, nothink-greedy 3.12, think-greedy 3.30 |
| `w1-graph-ab/` | the GDN a/b in-projection chunking: nsys node trace, greedy C1, transcripts | 18.9 / 30.3 / 42.6 / 33.0 / 17.9 tok/s; mma_gemv 128.5 ms/step, drafter 11.5, kda_recurrent 5.3, Kernel2 1.2 (was 4.5) |
| `w1-mtp5/` | the same-dispatch exactness gate (6 verify rows) vs w1-graph-ab | identical 4/4 |
| `sampled/w1-sampled`, `w1-argmax` | the drawn proposals (ratio verify) vs the argmax walk, same binary: MT-Bench think-sampled, timed_load C1 T=1, transcripts | MT-Bench 3.27 vs 2.80; T=1 16.9/29.3/37.2/31.3/17.1 vs 15.5/28.1/38.6/33.3/15.2; greedy transcripts identical 4/4 |
| `sampled/w4-sampled`, `w4-argmax` | four nodes, the same pair; benchy ran at its default tg32 (3 runs: noisy) | T=1 drawn 49.0/91.0/115.2/91.1/50.9 tok/s |
| `gemv/`, `ncu/` | the fp8 GEMV row-count curve (m = 1/4/8/16, cold L2) at worlds 1/2/4 before the staging change; Nsight Compute (root) at m = 1 and 8 | gate|up W1 228.5/220.8/220.0/209.5 GB/s; ncu: both ~490 us in isolation, 109 regs, 2 blocks/SM, 2.83 waves |
| `gemv2/`, `w1-pipelined/` | the register-pipelined activation staging: the curve again, transcripts (bitwise), greedy C1, MT-Bench with the truncated proposals | gate|up W1 227.1/222.2/219.4/227.0; W2 m=16 186 → 218; transcripts identical 4/4; 19.0/30.4/42.9/33.3/18.0 at 152 ms/step; MT-Bench 3.34 |
| `sampled/w1-temp0.7`, `w1-temp0.5` | the draft temperature as a fraction of the request's (timed_load T=1; MT-Bench at 0.7) | 0.7: 17.5/29.8/40.0/34.1/15.2, MT-Bench 3.41; 0.5: 16.2/32.0/42.8/32.5/15.8 |
| `sampled/w4-sampled-tg128`, `w4-argmax-tg128` | failed boots: the recipes carried `mtp_draft_temperature` before the binary parsed it | — |
| `sampled/w4-ship` | the four-node drafter recipe as it ships (drawn, 0.7): llama-benchy pp2048 tg128 x 20, timed_load T=1 | benchy 62.7 tok/s (std 5.9; the drafter read 50.1 and MTP depth 3 55.0 on 2026-10-04); T=1 52.2/84.4/117.0/98.2/47.6 |
| `sampled/w1-ship` | the one-node template's recipe: transcripts vs w1-graph-ab | identical 4/4 |
| `sampled/w4-argmax-tg128` (control) | the argmax walk under the same harness, same binary | benchy 53.6 tok/s (std 5.6): the drawn proposals +17 % |
| `ctest_rerun.log` | the host ctest suite with the device free (the first pass under production read 58 memory refusals) | see the log |

The arena-card figures are development yardsticks (not repository documentation): the
four-node card reads 58.48 tok/s on this harness.

#pragma once

#include <string>

namespace dgpp {

// Pins every page the process currently maps into RAM (mlockall
// MCL_CURRENT) and reports what happened.
//
// Why a serving process wants this: the decode loop's host side is a few
// hundred kilobytes of hot code and tables (tokenizer vocab, pick scratch,
// the bus engine's heap) sitting next to ~80 GB of resident model. When
// the box runs at its memory watermark the kernel swaps those cold-looking
// pages out, and the next step pays a 2-10 ms major fault for a vocab
// entry (2026-09-02: every ~10th token on the fabric, in lockstep across
// ranks because the token is shared). Locking after the model is loaded
// faults the swapped pages back once and forbids the next eviction.
//
// MCL_CURRENT only — MCL_FUTURE would turn every later allocation into a
// hard failure when RLIMIT_MEMLOCK is finite, and the hot set exists by
// the time this is called. The soft RLIMIT_MEMLOCK is raised to the hard
// one first (unprivileged, and ssh-spawned ranks arrive with a finite soft
// limit). Best effort: a hard limit below the process's current footprint
// or a missing CAP_IPC_LOCK makes the kernel refuse; the caller logs
// `*error` and keeps serving — locking is a latency measure, not a
// correctness one. Returns true when the lock is in place; `locked_bytes`
// (optional) receives the kernel's VmLck figure for the log line.
bool lock_process_memory(std::string* error, size_t* locked_bytes = nullptr);

// /proc/meminfo's MemAvailable in bytes (0 when unreadable): what the kernel
// will hand out once it reclaims the page cache — which cudaMemGetInfo's
// "free" on the GB10's unified pool does not count (after one resident load
// the checkpoint's file pages sit in that cache, reclaimable).
size_t host_memory_available_bytes();

}  // namespace dgpp

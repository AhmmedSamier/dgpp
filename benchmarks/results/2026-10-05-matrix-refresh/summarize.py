#!/usr/bin/env python3
"""Render overview tables from completed measurement groups only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CLASSES = ["prose", "code", "json", "math", "chat"]
LENGTHS = [0, 32768, 65536, 131072, 262144, 524288]
MODES = ["plain", "mtp1", "mtp2", "mtp3", "mtp4", "dspark1", "dspark2",
         "dspark3", "dspark5", "dspark-adaptive", "dflash2", "dflash2-fixed"]
NAMES = ["Plain", "MTP1", "MTP2", "MTP3", "MTP4", "DSpark 1", "DSpark 2",
         "DSpark 3", "DSpark 5", "DSpark adaptive", "DFlash2 adaptive", "DFlash2 fixed 7"]


def table(headers, rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |"
                     for row in [headers, ["---"] * len(headers), *rows])


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def completed(directory, name):
    receipt = read(directory / f"{name}.log.command.json")
    memory = read(directory / 'memory-check.json')
    identity = read(directory / 'rank-identity.json')
    shutdown = read(directory / 'down.log.command.json')
    return (read(directory / f"{name}.json")
            if receipt and receipt.get("returncode") == 0 and memory and memory.get('passed')
            and identity and identity.get('passed') and shutdown and shutdown.get('returncode') == 0
            else None)


def class_rates(report, concurrency, engine=True):
    if report is None:
        return None
    groups = defaultdict(list)
    for phase in report["phases"]:
        if phase["concurrency"] == concurrency:
            value = phase["engine"]["tokens_per_s"] if engine else phase["metrics"]["wall_tokens_per_s"]
            groups[phase["class"]].append(value)
    if set(groups) != set(CLASSES) or any(len(values) != 3 for values in groups.values()):
        return None
    return [statistics.median(groups[c]) for c in CLASSES]


def span(values):
    if values is None:
        return "Pending"
    lo, hi = f"{min(values):.1f}", f"{max(values):.1f}"
    return lo if lo == hi else f"{lo}–{hi}"


def context_groups(report):
    groups = defaultdict(list)
    if report:
        for sample in report["samples"]:
            groups[sample["requested_tokens"]].append(sample)
    return {length: samples for length, samples in groups.items()
            if sorted(s["repeat"] for s in samples) == [0, 1, 2]}


def generate():
    matrix = read(HERE / "matrix.json")
    serving, classes, modes, contexts, prefills, config_rows, coverage = [], [], [], [], [], [], []
    detail = ["# Benchmark matrix refresh", "",
              "All throughput values are medians of three repetitions. Serving ranges span the five prompt classes.",
              "The [manifest](manifest.json) identifies the engine revision and binary. Commands, exit codes, configurations and responses are retained under `raw/`.", ""]
    detail += ["Only runs with passing per-rank process-swap checks contribute values. "
               "The [initial attempt](invalid-swapping/INVALID.md) is invalid; "
               "the [memory investigation](diagnostics/README.md) records the cause and fix.", ""]
    missing = []
    for entry in matrix:
        ident = entry["id"]
        cfg = read(HERE / entry["config"])
        engine = cfg["engine"]
        directory = HERE / "raw" / ident / "refresh"
        greedy = completed(directory / "default", "greedy")
        context_reports = [completed(directory / "default", f"context-{length}") for length in LENGTHS]
        context_reports = [report for report in context_reports if report is not None]
        context = ({"surface": context_reports[0]["surface"],
                    "samples": [s for report in context_reports for s in report["samples"]]}
                   if context_reports else None)
        prefill = completed(directory / "default", "prefill")
        key = [entry["label"], entry["world"], entry["options"]]
        default_label = NAMES[MODES.index(entry["default_mode"])]
        if entry["default_mode"] == "dspark-adaptive":
            default_label += f" ≤{engine['mtp_depth']}"
        elif entry["default_mode"] == "dflash2":
            default_label += " ≤7"
        rates = class_rates(greedy, 1)
        prefill_values = []
        for length in [2048, 8192, 32768]:
            samples = [s for s in (prefill or {}).get("samples", []) if s.get("requested_tokens") == length]
            prefill_values.append(f"{statistics.median(s['prefill_ms'] for s in samples) / 1000:.3f}"
                                  if len(samples) == 3 else "Pending")
        serving.append(key + [span(rates)] + [span(class_rates(greedy, c, False)) for c in [1, 2, 4, 8]]
                       + [" / ".join(prefill_values)])
        classes.append(key + ([f"{v:.1f}" for v in rates] if rates else ["Pending"] * 5))
        row = key + [default_label]
        done_modes = 0
        for mode in MODES:
            if mode == entry["default_mode"]:
                values = rates
            elif mode in entry["modes"]:
                values = class_rates(completed(directory / mode, "greedy"), 1)
            else:
                row.append("Not swept")
                continue
            row.append(span(values))
            if values is None:
                missing.append(f"{ident}/{mode}")
            else:
                done_modes += 1
        modes.append(row)
        groups = context_groups(context)
        limit = (context or {}).get("surface", {}).get("context", {}).get("request_limit_tokens", engine["kv_capacity"])
        cold32 = groups.get(32768)
        cold32 = next((s["engine"]["prefill_ms"] / 1000 for s in cold32 if s["repeat"] == 0), None) if cold32 else None
        row = key + [f"{limit:,}", f"{cold32:.3f}" if cold32 is not None else "Pending"]
        cold = key.copy()
        for length in LENGTHS:
            if length > limit:
                row += ["Limit", "Limit"]
                cold.append("Limit")
            elif length not in groups:
                row += ["Pending", "Pending"]
                cold.append("Pending")
                missing.append(f"{ident}/context/{length}")
            else:
                samples = groups[length]
                row += [f"{statistics.median(s['ms_per_pass'] for s in samples):.2f}",
                        f"{statistics.median(s['engine_tokens_per_s'] for s in samples):.1f}"]
                cold.append(f"{next(s['engine']['prefill_ms'] for s in samples if s['repeat'] == 0) / 1000:.3f}")
        contexts.append(row)
        prefills.append(cold)
        kv_format = engine.get("kv_dtype", "bf16")
        if entry["model"].startswith("deepseek-ai/DeepSeek-V4.1"):
            kv_format = "FP4 blocks / FP8 window"
        elif entry["model"].startswith("deepseek-ai/DeepSeek-V4-"):
            kv_format = "BF16 (quantized values)"
        config_rows.append(key + [engine["max_concurrency"], f"{engine['kv_capacity']:,}",
                                  kv_format,
                                  engine.get("prefix_cache_gib", 0),
                                  default_label,
                                  f"[JSON](../benchmarks/results/{HERE.name}/{entry['config']})"])
        coverage.append([ident, "Complete" if greedy else "Pending", f"{done_modes}/{len(entry['modes']) + 1}",
                         f"{len(groups)}/{sum(n <= limit for n in LENGTHS)}", "Complete" if prefill else "Pending"])
        if not greedy:
            missing.append(f"{ident}/serving")
        if not prefill:
            missing.append(f"{ident}/prefill")
        detail += [f"## {ident}", "", f"Configuration: [{entry['config']}]({entry['config']}). "
                   f"Default mode: {entry['default_mode']}. Raw records: [raw/{ident}/refresh/](raw/{ident}/refresh/).", ""]
        if greedy:
            detail_rows = []
            for c in [1, 2, 4, 8]:
                for cls, eng, wall in zip(CLASSES, class_rates(greedy, c), class_rates(greedy, c, False)):
                    detail_rows.append([cls, c, f"{eng:.2f}", f"{wall:.2f}"])
            detail += [table(["Class", "C", "Engine tok/s", "Wall tok/s"], detail_rows), ""]
        if groups:
            detail_rows = []
            for length, samples in sorted(groups.items()):
                first = next(s for s in samples if s["repeat"] == 0)
                detail_rows.append([length, first["answer"]["usage"]["prompt_tokens"],
                    f"{first['engine']['prefill_ms'] / 1000:.3f}",
                    f"{statistics.median(s['ms_per_pass'] for s in samples):.2f}",
                    f"{statistics.median(s['tokens_per_pass'] for s in samples):.2f}",
                    f"{statistics.median(s['engine_tokens_per_s'] for s in samples):.2f}"])
            detail += [table(["Context bucket", "Actual prompt tokens", "Cold prefill s", "ms/pass", "Tokens/pass", "Engine tok/s"], detail_rows), ""]
        snapshots = sorted((directory / "default").glob("context-*-health.json"))
        if snapshots:
            detail += ["Runtime health snapshots: " + ", ".join(
                f"[{path.stem.removeprefix('context-').removesuffix('-health')}]({path.relative_to(HERE)})"
                for path in snapshots) + ".", ""]
    key_headers = ["Model / weights", "Nodes", "Options"]
    mode_tables = []
    for title, predicate, selected, unavailable in [
        ("Native MTP", lambda e: not e['default_mode'].startswith(('dspark', 'dflash'))
         and not e['model'].startswith('XiaomiMiMo/'),
         ["plain", "mtp1", "mtp2", "mtp3", "mtp4"], {}),
        ("MiMo: MTP and DFlash", lambda e: e['model'].startswith('XiaomiMiMo/'),
         ["plain", "mtp1", "mtp2", "mtp3"], {"DFlash": "Not loaded by engine"}),
        ("DeepSeek DSpark", lambda e: e['default_mode'].startswith('dspark'),
         ["plain", "dspark1", "dspark2", "dspark3", "dspark5", "dspark-adaptive"], {}),
        ("Qwen 27B: MTP and DFlash2", lambda e: e['default_mode'].startswith('dflash'),
         ["plain", "mtp1", "mtp2", "mtp3", "dflash2-fixed", "dflash2"], {}),
    ]:
        selected_rows = [row[:4] + [row[4 + MODES.index(mode)] for mode in selected] + list(unavailable.values())
                         for entry, row in zip(matrix, modes) if predicate(entry)]
        mode_tables += [f"### {title}", "",
                        table(key_headers + ["Default"] + [NAMES[MODES.index(m)] for m in selected]
                              + list(unavailable), selected_rows), ""]
    tables = {
        "serving": table(key_headers + ["C1 engine tok/s", "C1 wall tok/s", "C2 wall tok/s", "C4 wall tok/s", "C8 wall tok/s", "Cold prefill s: ~2K / ~8K / ~32K"], serving),
        "classes": table(key_headers + [c.upper() if c == "json" else c.title() for c in CLASSES], classes),
        "config": table(key_headers + ["Slots", "KV pool tokens", "KV format", "Prefix cache GiB", "Default decode", "Configuration"], config_rows),
        "modes": "\n".join(mode_tables).rstrip(),
        "context": table(key_headers + ["Context length (limit)", "Cold prefill (s), 32K"] +
                         [title for n in ["0", "32K", "64K", "128K", "256K", "512K"]
                          for title in [f"Decode ms/pass ({n})", f"Engine tok/s ({n})"]], contexts),
        "context-prefill": table(key_headers + ["0", "32K", "64K", "128K", "256K", "512K"], prefills),
    }
    overview = ROOT / "docs/benchmarks.md"
    text = overview.read_text()
    for name, rendered in tables.items():
        start, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
        if start not in text or end not in text:
            raise ValueError(f"overview missing markers for {name}")
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + start + "\n\n" + rendered + "\n\n" + end + after
    overview.write_text(text)
    detail[4:4] = [f"Outstanding measurement groups: {len(missing)}.", "",
                   table(["Deployment", "C1/C2/C4/C8", "Decode modes", "Context buckets", "Cold prefill"], coverage), ""]
    (HERE / "README.md").write_text("\n".join(detail).rstrip() + "\n")
    (HERE / "coverage.json").write_text(json.dumps({"missing": missing, "deployments": coverage}, indent=2) + "\n")
    return missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    missing = generate()
    print(f"Rendered overview; {len(missing)} outstanding measurement groups")
    if args.require_complete and missing:
        raise SystemExit(1)

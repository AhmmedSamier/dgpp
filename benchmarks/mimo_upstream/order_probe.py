"""Diagnose arrival-order sensitivity; timings here are not benchmark samples."""

import argparse
import concurrent.futures
import json
import time
from benchmark import ROOT, PROMPTS, call, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument(
        "--logprobs",
        action="store_true",
        help="Record top-2 probabilities; diagnostic only",
    )
    args = parser.parse_args()
    assert args.repeats > 0 and args.delay >= 0
    dest = ROOT / "raw" / args.tag / "order-probe.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert not dest.exists(), "Preserve previous evidence"
    before = metrics()
    assert before["scheduler"]["active"] == before["scheduler"]["queued"] == 0
    records = []

    def save(result):
        records.append(result)
        dest.write_text(json.dumps({"before": before, "records": records}, indent=2))

    for rep in range(args.repeats):
        for name in ("code", "json"):
            save(call(f"{name}-solo-{rep}", PROMPTS[name], 256, False, args.logprobs))
        for order in [("code", "json"), ("json", "code")]:
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                first = pool.submit(
                    call,
                    f"{order[0]}-first-{rep}",
                    PROMPTS[order[0]],
                    256,
                    False,
                    args.logprobs,
                )
                time.sleep(args.delay)
                second = pool.submit(
                    call,
                    f"{order[1]}-second-{rep}",
                    PROMPTS[order[1]],
                    256,
                    False,
                    args.logprobs,
                )
                save(first.result())
                save(second.result())
    after = metrics()
    assert after["service"]["requests_total"] - before["service"][
        "requests_total"
    ] == len(records), "external traffic"
    assert after["service"]["requests_failed"] == before["service"]["requests_failed"]
    assert not after["service"]["engine_failed"]
    assert after["scheduler"]["active"] == after["scheduler"]["queued"] == 0
    dest.write_text(
        json.dumps({"before": before, "records": records, "after": after}, indent=2)
    )
    print("PASS accounting; output equivalence must be assessed separately")


if __name__ == "__main__":
    main()

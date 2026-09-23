"""Real serving branch/reuse validation; generated text is never executed."""

import pathlib, sys, json
from benchmark import ROOT, call, metrics

R = ROOT
tag = sys.argv[1]
out = R / "raw" / tag / "cache-reuse.json"
assert not out.exists()
prefix = "Shared-prefix branch validation.\n" + "".join(
    f"Record {i}: value {i%17}.\n" for i in range(700)
)
words = ["cedar", "amber", "willow", "raven"]
before = metrics()
records = []
for word in words:
    for round in range(2):
        prompt = (
            prefix
            + "\nBranch marker "
            + word
            + ". Ignore the records for your answer. Reply with exactly the branch marker, one lowercase word and nothing else."
        )
        r = call(f"branch-{round}-{word}", prompt, 128, True)
        r["expected"] = word
        r["metrics_after"] = metrics()
        records.append(r)
        out.write_text(json.dumps({"before": before, "records": records}, indent=2))
        assert r["content"].strip() == word, (word, r["content"], r["reasoning"])
        if round == 1:
            assert (
                r["usage"]["prompt_tokens_details"]["cached_tokens"] > 7000
            ), "repeat did not exercise restore"
after = metrics()
assert after["service"]["requests_total"] - before["service"]["requests_total"] == 8
assert (
    not after["service"]["engine_failed"]
    and after["service"]["requests_failed"] == before["service"]["requests_failed"]
)
assert after["prefix_cache"]["enabled"]
out.write_text(
    json.dumps({"before": before, "records": records, "after": after}, indent=2)
)
print(
    "PASS branch/reuse8; owned bytes",
    before["prefix_cache"].get("snapshot_storage_bytes"),
    after["prefix_cache"].get("snapshot_storage_bytes"),
    flush=True,
)

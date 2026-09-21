"""Exercise both scorer modes and the analyzer on the native Qwen fixture."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qwen_head_compare import compare


def main():
    binary, checkpoint = map(lambda x: str(Path(x).resolve()), sys.argv[1:])
    with tempfile.TemporaryDirectory(prefix="qwen-head-fixture-") as directory:
        root = Path(directory)
        manifest = root / "corpora.json"
        manifest.write_text(json.dumps([{"name": "fixture", "ids": [5 + i % 100 for i in range(256)]}]))
        for mode in ("gemv", "mma"):
            (root / mode).mkdir()
            with (root / mode / "r0.log").open("w") as log:
                subprocess.run([binary, "--checkpoint-dir", checkpoint, "--requests", str(manifest),
                                "--fp8-head", mode, "--boundary-tokens", "128", "--prefill-trials", "2"],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        try:
            result = compare(str(root / "gemv"), str(root / "mma"))
        except Exception:
            for mode in ("gemv", "mma"):
                retained = Path(tempfile.mkdtemp(prefix=f"qwen-head-failed-{mode}-"))
                (retained / "r0.log").write_bytes((root / mode / "r0.log").read_bytes())
                print(f"Retained {mode} diagnostic log at {retained}", flush=True)
            raise
        if not result["passed"]:
            raise RuntimeError(json.dumps(result))
        print(f"Both scorer modes, all {len(result['cases'])} cases, and exact repeat controls passed.")


if __name__ == "__main__":
    main()

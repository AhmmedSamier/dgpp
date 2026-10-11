"""Final paired-prefill validation, one freshly deployed world at a time."""
import datetime, hashlib, json, os, pathlib, subprocess
out = pathlib.Path(__file__).parent
root = pathlib.Path('/home/stephen/workspace/dgpp')
binary = out / 'dgpp-serve-prefill-pair-final'
assert binary.exists()
controller = out / 'prefill_final_campaign.py'
status = []
for world in (2, 1, 4):
    for suffix, quant in (('fp8a', 'fp8'), ('nvfp4', 'nvfp4'), ('fp8b', 'fp8')):
        label = f'prefill-pair-final-w{world}-{suffix}'
        cmd = ['python3', str(controller), quant, str(world), label, '--prefill-only']
        # Recheck complete longer-context transcripts on the changed prefill.
        if world == 4:
            cmd.append('--decode-context')
        receipt = dict(label=label, start=datetime.datetime.now(datetime.timezone.utc).isoformat(), command=cmd)
        print(receipt['start'], label, flush=True)
        with (out / (label + '-controller.log')).open('w') as log:
            p = subprocess.run(cmd, cwd=root, env={**os.environ, 'PR95_BIN': str(binary)},
                               stdout=log, stderr=subprocess.STDOUT, timeout=2400)
        receipt.update(returncode=p.returncode, end=datetime.datetime.now(datetime.timezone.utc).isoformat())
        status.append(receipt)
        (out / 'prefill-pair-final-status.json').write_text(json.dumps(status, indent=2) + '\n')
        assert p.returncode == 0, receipt
    subprocess.run(['python3', str(out / 'audit_pair_final.py'), str(world)], check=True)
subprocess.run(['python3', str(out / 'audit_pair_final.py')], check=True)

subprocess.run(['python3', str(out / 'audit_pair_context.py')], check=True)
subprocess.run(['python3', str(out / 'audit_release_combined.py')], check=True)

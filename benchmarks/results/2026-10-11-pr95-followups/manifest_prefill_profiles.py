import datetime,hashlib,json,pathlib,subprocess
out=pathlib.Path(__file__).parent
rows=[]
for prefix in ('prefill-gap-profile','prefill-tail-profile'):
    for quant in ('fp8','nvfp4'):
        folder=out/f'{prefix}-{quant}'
        binary_sha=(folder/'binary.sha256').read_text().strip()
        for p in sorted(folder.iterdir()):
            if p.suffix not in ('.nsys-rep','.sqlite'):continue
            with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            rows.append(dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=digest,binary_sha256=binary_sha))
for p in sorted(out.glob('prefill-large-*.ncu-rep')):
    with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    rows.append(dict(path=p.name,bytes=p.stat().st_size,sha256=digest))
result=dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),local_root=str(out),profiles=rows,
            nsys_version=subprocess.check_output(['/usr/local/cuda/bin/nsys','--version'],text=True).strip(),ncu_version=subprocess.check_output(['/usr/local/cuda/bin/ncu','--version'],text=True).strip(),
            note='Raw captures and exported databases remain local. Gap exploratory prompts differ from the final matched prompts. Tail profiles use the original matched prompts at f5a517c1 on both TP2 ranks. Large NCU reports are synthetic microprobes with application replay, not serving performance results.')
(out/'prefill-raw-profile-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(f'Manifested {len(rows)} raw profile files')

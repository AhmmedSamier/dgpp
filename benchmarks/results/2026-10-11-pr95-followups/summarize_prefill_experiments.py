from pathlib import Path
import json,re,statistics,hashlib
out=Path(__file__).parent
summary={'method':'Synthetic random inputs; per-output bitwise comparisons with current production. Microbenchmark timings are selection evidence, not end-to-end serving performance. Brackets where present use the faster baseline.'}
for label,file in [('conversion','prefill-convert-probe.log'),('instruction_schedule','prefill-schedule-probe.log'),('aspect_ratio','prefill-aspect-probe.log'),('swiglu','prefill-swiglu-probe.log'),('swiglu8','prefill-swiglu8-probe.log'),('swiglu_lut','prefill-swiglu-lut-probe.log'),('pair_distinct','prefill-pair-distinct.log'),('pair_warp','prefill-pair-warp-probe.log')]:
 groups={};p=out/file
 for line in p.read_text().splitlines():
  if 'mismatch=' not in line:continue
  fields=dict(re.findall(r'([A-Za-z_]+)=([^\s]+)',line));variant=fields.get('variant','default')
  if variant[0].isdigit():variant=re.search(r'variant=(\w+)',line).group(1)
  key=(variant,int(fields['world']),int(fields.get('down',0)),int(fields['m']))
  groups.setdefault(key,[]).append(fields)
 rows=[]
 for (variant,world,down,m),rs in groups.items():
  rows.append({'variant':variant,'world':world,'down':bool(down),'rows':m,'repetitions':len(rs),'median_change_pct':statistics.median(float(r['change'].rstrip('%')) for r in rs),'mismatches':sum(int(r['mismatch']) for r in rs)})
 summary[label]={'source_file':file,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'results':rows}
summary['decisions']={'conversion':'Rejected: slower or no consistent gain.','instruction_schedule':'Rejected: no consistent improvement in long TP2 shapes. Guarded reciprocal not integrated.','aspect_ratio':'Rejected: K32 variants slower.','medium_rows':'Rejected: prefill-tail-probe-summary.json shows no consistent useful advantage.','swiglu8':'Rejected after serving trial: despite 2.17–3.05% micro gains, 8K/32K were 0.47%/0.57% slower than the stronger same-format baseline. See prefill-fusion-trial-audit.json.','swiglu_lut':'Rejected: exact BF16-input SiLU table did not improve the needed shape.'}
summary['decisions'].update(pair_distinct='Shared-memory epilogue exchange: all54 distinct-weight samples bitwise; serving8K/32K improved0.33%/0.38%. Superseded by same-warp form.',pair_warp='Selected: each warp computes matching gate and up columns; avoids epilogue block exchange. All27 distinct-weight samples bitwise. TP2serving8K/32K improves0.953%/0.662% against stronger baseline; 512-row control dispatch unchanged. Register128,16-byte stack with12-byte spills; retained on end-to-end evidence. Final FP8 brackets are recorded separately.',budget='Rejected before measurements: Qwen35 lacks kResumablePrefill, so8192 idle budget cannot be enabled by merely changing model chunk capacity. Private source reverted.')
(out/'prefill-experiments-summary.json').write_text(json.dumps(summary,indent=2)+'\n')

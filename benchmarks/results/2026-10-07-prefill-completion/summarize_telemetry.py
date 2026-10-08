"""Summarize local telemetry without retaining the per-second captures in Git."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
result={'scope':'All samples in each validated launch, including startup and shutdown. GPU states are sampled approximately every five seconds; an inactive sample does not rule out an event between samples. Host UTC clocks differ; timestamps below use each node\'s own clock.','launches':{}}
for record in sorted((HERE/'raw').glob('*/refresh/*/record.json')):
 launch={}
 for path in sorted(record.parent.glob('node*-telemetry.jsonl')):
  data=path.read_bytes(); rows=[json.loads(line) for line in data.decode().splitlines()]
  memory=[r for r in rows if r.get('kind')=='memory' and r.get('serving_processes')]
  gpu=[r for r in rows if r.get('kind')=='gpu' and r.get('returncode')==0]
  parsed=[]
  for r in gpu:
   fields=[f.strip() for f in r['data'].split(',')]
   parsed.append({'at':r['at'],'temperature_c':int(fields[0]),'power_w':float(fields[1].split()[0]),'sm_mhz':int(fields[2].split()[0]),'software_thermal_slowdown':fields[3],'hardware_thermal_slowdown':fields[4]})
  launch[path.name]={'sha256':hashlib.sha256(data).hexdigest(),'serving_memory_samples':len(memory),'minimum_available_gib':min(r['MemAvailable_KiB']/2**20 for r in memory),'gpu_samples':len(parsed),'maximum_temperature_c':max(r['temperature_c'] for r in parsed),'sm_mhz_range':[min(r['sm_mhz'] for r in parsed),max(r['sm_mhz'] for r in parsed)],'thermal_slowdown_samples':[r for r in parsed if r['software_thermal_slowdown']=='Active' or r['hardware_thermal_slowdown']=='Active']}
 result['launches'][str(record.parent.relative_to(HERE))]=launch
(HERE/'telemetry-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('Summarized telemetry for',len(result['launches']),'validated launches')

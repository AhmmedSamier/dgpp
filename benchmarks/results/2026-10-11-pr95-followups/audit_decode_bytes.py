import collections,json,pathlib,re,struct
out=pathlib.Path(__file__).parent;hub=pathlib.Path.home()/'.cache/huggingface/hub';report={'method':'Checkpoint tensor headers, language transformer only. Nominal TP4 divides shardable tensors by four; small replicated norms/global scales and runtime FP32 scale widening are explicitly excluded from this approximation. These are useful source bytes, not measured DRAM traffic.','models':[]}
for quant,repo,rev in [('fp8','Qwen/Qwen3.8-27B-FP8','017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'),('nvfp4','unsloth/Qwen3.8-27B-NVFP4','f0b7c9e722f5565102fff8481c99e4d86ae099c7')]:
 d=hub/('models--'+repo.replace('/','--'))/'snapshots'/rev;tensors=[];totals=collections.Counter();dtypes=collections.Counter();layer_formats=collections.defaultdict(set)
 for f in sorted(d.glob('*.safetensors')):
  with f.open('rb') as x:n=struct.unpack('<Q',x.read(8))[0];h=json.loads(x.read(n))
  for name,v in h.items():
   if name=='__metadata__':continue
   m=re.match(r'model.language_model.layers.(\d+)\.(.*)',name)
   if not m:continue
   layer=int(m[1]);suffix=m[2];group=('first56_mlp' if layer<56 else 'last8_mlp') if suffix.startswith('mlp.') else 'attention_and_norms';size=v['data_offsets'][1]-v['data_offsets'][0];totals[group]+=size;dtypes[v['dtype']]+=size
   if suffix.startswith('mlp.') and (suffix.endswith('weight') or suffix.endswith('weight_packed')):layer_formats[layer].add(v['dtype'])
   tensors.append(dict(name=name,group=group,dtype=v['dtype'],shape=v['shape'],bytes=size,shard=f.name))
 report['models'].append(dict(quant=quant,repo=repo,revision=rev,total_transformer_source_bytes=sum(totals.values()),nominal_source_bytes_per_rank={k:v/4 for k,v in totals.items()},dtypes=dict(dtypes),mlp_layer_formats={k:sorted(v) for k,v in layer_formats.items()},tensors=tensors))
a,b=report['models'];report['transformer_source_byte_reduction']=1-b['total_transformer_source_bytes']/a['total_transformer_source_bytes'];report['tp4_fp8_head_payload_bytes']=248320//4*5120;report['fp4_payload_plus_group_scale_fraction_of_fp8_payload']=9/16
(out/'decode-byte-audit.json').write_text(json.dumps(report,indent=2));print([(m['quant'],m['nominal_source_bytes_per_rank']) for m in report['models']]);print('reduction',report['transformer_source_byte_reduction'])

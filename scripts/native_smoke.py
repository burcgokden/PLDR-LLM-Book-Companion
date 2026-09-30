#!/usr/bin/env python3
"""Native cache, restoration and optimizer-state replay smoke. Apache-2.0."""
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,json,sys,time,struct
import torch
import argparse
parser=argparse.ArgumentParser(description='Bounded native smoke on synthetic tokens, not a scientific acquisition.')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--device',default='cuda:0',choices=['cuda:0','cuda:1','cpu'])
args=parser.parse_args()
R=Path(__file__).resolve().parents[1];D=R/'companions/dynamics'
if args.output.exists(): raise FileExistsError('Use a fresh output path')
args.output.parent.mkdir(parents=True,exist_ok=True)
sys.dont_write_bytecode=True
sys.path[:0]=[str(D),str(D/'vendor/model/src'),str(D/'vendor/model/scripts')]
from model_rg.training import TrainingModel
from model_rg.variance_family import normalize_variance_initialization
from model_rg.criticality import optimizer_for, fix_shared_generator, parameter_digest
from model_rg.inference_interventions import operator_cache,fixed_operators,projected_rows
from model_rg.run_outcome import clip_finite_norm
from model_rg import native_state_schema as schema
from model_rg.replay_equality import require_replay
from run_critical_onepass import observe

def cpu_tree(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu_tree(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
 return x
native_binding=schema.registered_binding([2,4])
schema.validate_binding(
 {'design':{'heads':[2,4]},'native_state_contract':native_binding},
 {'native_model':D/'vendor/native/PLDR-LLM-v51-SOC-110M-1/modeling_pldrllm.py',
  'native_config':D/'vendor/native/PLDR-LLM-v51-SOC-110M-1/configuration_pldrllm.py'},
 D/'vendor/model')

start=time.monotonic();torch.set_num_threads(2)
if args.device.startswith('cuda'):
 if not torch.cuda.is_available():raise RuntimeError('A compatible CUDA device is required')
 torch.cuda.set_device(args.device)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(True)
if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
rng_checks=[]
if args.device.startswith('cuda'):
 for seed,offset in [(0,0),(2**64-1,4),(17,2**63),(3,2**64-4)]:
  state=torch.tensor(list(struct.pack('<QQ',seed,offset)),dtype=torch.uint8)
  schema.validate_cuda_rng(state)
  a=torch.Generator(device=args.device);b=torch.Generator(device=args.device)
  a.set_state(state);b.set_state(state)
  assert torch.equal(a.get_state(),state)
  assert torch.equal(torch.rand(257,device=args.device,generator=a),torch.rand(257,device=args.device,generator=b))
  assert torch.equal(a.get_state(),b.get_state())
  rng_checks.append(dict(seed=seed,offset=offset,restored=True,continuation_equal=True))
 for label,state in [('one_byte',torch.zeros(1,dtype=torch.uint8)),('unaligned_offset',torch.tensor(list(struct.pack('<QQ',7,1)),dtype=torch.uint8))]:
  try:schema.validate_cuda_rng(state)
  except ValueError:pass
  else:raise AssertionError('CPU CUDA parser accepted '+label)
  try:torch.Generator(device=args.device).set_state(state)
  except RuntimeError:pass
  else:raise AssertionError('Native CUDA setter accepted '+label)
  rng_checks.append(dict(control=label,cpu_parser_rejected=True,native_setter_rejected=True))
records=[]
for heads in [2,4]:
 model=TrainingModel(D/'vendor/native/PLDR-LLM-v51-SOC-110M-1',heads,62509,args.device)
 normalize_variance_initialization(model);shared=fix_shared_generator(model,2031)
 declared=schema.model_schema(heads)
 assert list(model.model.state_dict())==list(declared)==[n for n,p in model.model.named_parameters()]
 assert all(list(model.model.state_dict()[n].shape)==v['shape'] for n,v in declared.items())
 rng=torch.Generator(device='cpu').manual_seed(321)
 batch=torch.randint(1,model.config.vocab_size,(2,65),generator=rng).to(model.device)
 model.model.eval();calls=[0]
 def hook(*args): calls[0]+=1
 hooks=[unit.register_forward_hook(hook) for layer in model.model.decoder.dec_layers for unit in layer.mha1.reslayerAs]
 with torch.no_grad():
  output=model.forward(batch[:,:64],capture=True);z=output.logits[:,-1].clone();cache=operator_cache(output);del output
  native=calls[0];calls[0]=0
  with fixed_operators(model,cache):cached=model.forward(batch[:,:64]).logits[:,-1].clone()
  bypass=calls[0];assert bypass==0 and torch.equal(z,cached)
  restored=model.forward(batch[:,:64]).logits[:,-1].clone();assert torch.equal(z,restored)
  changed=batch[:,:64].clone();changed[:,32:]=(changed[:,32:]% (model.config.vocab_size-1))+1
  full=model.model(batch[:,:64],use_cache=False,logits_to_keep=0).logits
  alternate=model.model(changed,use_cache=False,logits_to_keep=0).logits
  first=float((full[:,0]-alternate[:,0]).abs().max())
  history=float((full[:,1:32]-alternate[:,1:32]).abs().max())
  with projected_rows(model):proj=model.forward(batch[:,:64]).logits[:,-1]
  assert torch.isfinite(proj).all()
 for h in hooks:h.remove()
 optimizer=optimizer_for(model,2.0);model.model.train()
 def step():
  model.model.train()
  optimizer.zero_grad(set_to_none=True)
  logits=model.forward(batch[:,:64]).logits[:,-1]
  loss=torch.nn.functional.cross_entropy(logits,batch[:,64]);assert torch.isfinite(loss)
  loss.backward();norm=clip_finite_norm(list(model.model.parameters()));optimizer.step()
  assert all(torch.isfinite(p).all() for p in model.model.parameters())
  return float(loss.detach()),float(norm),parameter_digest(model)
 # Warm up once, then serialize a complete state with nonempty moments.
 step()
 probes=torch.randint(1,model.config.vocab_size,(128,65),generator=rng).to(model.device)
 def snapshot(steps):
  return dict(model=cpu_tree(model.model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),
    step=steps,cpu_rng=torch.get_rng_state().clone(),
    cuda_rng=torch.cuda.get_rng_state(args.device).clone() if args.device.startswith('cuda') else torch.tensor(list(struct.pack('<QQ',7,0)),dtype=torch.uint8))
 initial=snapshot(1);schema.validate_checkpoint(initial,heads,2.,1)
 # Exercise the same safe deserializer as admission, not just an in-memory copy.
 import io
 serialized=io.BytesIO();torch.save(initial,serialized);serialized.seek(0)
 saved=torch.load(serialized,map_location='cpu',weights_only=True)
 loss,norm,digest=step();observed=observe(model,probes,True);endpoint=snapshot(2)
 schema.validate_checkpoint(endpoint,heads,2.,2)
 model.model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
 torch.set_rng_state(saved['cpu_rng'])
 if args.device.startswith('cuda'):torch.cuda.set_rng_state(saved['cuda_rng'],args.device)
 loss2,norm2,digest2=step();observed2=observe(model,probes,True);endpoint2=snapshot(2)
 require_replay(endpoint,endpoint2);require_replay(observed,observed2)
 assert digest2==digest and loss2==loss and norm2==norm
 rec=dict(heads=heads,layers=5,metric_units_per_layer=8,batch_size=2,prefix_length=64,synthetic_tokens=True,native_generator_calls=native,cached_generator_calls=bypass,own_cache_bitwise_equal=True,restored_bitwise_equal=True,first_row_suffix_change_max=first,historical_rows_suffix_change_max=history,loss=loss,gradient_norm=norm,one_step_replay_identical=True,nonempty_optimizer_restored=True,shared_digest=shared,schema_keys=len(declared),all_parameter_moments_populated=len(initial["optimizer"]["state"])==len(declared),cpu_rng_restored=True,cuda_rng_restored=args.device.startswith("cuda"),optimizer_endpoint_equal=True,rng_endpoints_equal=True,observations_equal=True,serialized_state_restored=True)
 records.append(rec);print(rec,flush=True)
 del model,optimizer,initial,saved,endpoint,endpoint2,serialized,observed,observed2,z,cache,cached,restored,full,alternate,batch,proj,probes
 torch.cuda.empty_cache()
report=dict(status='passed',scope='Synthetic bounded native-program smoke: two widths, one warm-up AdamW step then one step replayed from a serialized complete model/Adam/CPU-CUDA RNG state per width, with all moments, random endpoints and observations compared; no scientific training campaign or checkpoint-quality replication.',records=records,rng_format_checks=rng_checks,native_state_schema=schema.SCHEMA,native_state_binding=native_binding,runtime_binding=schema.RUNTIME,qualification_updates=0,scientific_updates=0,smoke_optimizer_updates=6,elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,torch=torch.__version__,device_type=torch.device(args.device).type)
args.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

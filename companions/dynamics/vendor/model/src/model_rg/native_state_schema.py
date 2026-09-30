"""Registered CPU contract for the learned-operator TrainingModel family.

Schema discovery never imports or constructs a native model. Only the pinned
configuration (including its true bias, normalization and cache branches) is
supported. Tests compare these declarations with independently constructed
native state_dict/named_parameters outputs. Values can be synthetic: semantic
admissibility and replay equality do not prove historical execution.
"""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import sys

import numpy as np
import torch

SCHEMA = 'pldr-native-training-state-v1'
VOCABULARY = 32000
# These are registration anchors, not digests supplied by the candidate state.
NATIVE_SOURCES = {
    'native_model': '34b65af6b73a384fca547b7000e61cbb995e537246e196ea4d9e803fd92031c2',
    'native_config': '8dd802d7bf00e024446f15eaea03abfb2e4834065858a28a898a201b2019cc2e',
}
ADAPTER_SOURCES = {
    'src/model_rg/training.py': '04157d038de4febc1aafdfa2593532dd4cb6b0d04ceee939b42b41fcba0e4922',
    'src/model_rg/native.py': '2b468292b0e823303c88affb88c2f8f895fa7c1835e8ee57e3f448b40c2162d0',
    'src/model_rg/criticality.py': '7c3f459226d1b2f83a46e637b0adf05b9534dbf88c6895b323b8fdc231868099',
}
RUNTIME = dict(torch='2.12.1+cu132', torch_git='7269437d655783a26cba32aa88195b741ff496aa',
               cuda='13.2', transformers='5.12.1', byteorder='little', pointer_bits=64)
RNG = dict(cpu='torch-CPUGeneratorImpl-current', cuda='torch-CUDAGeneratorImpl-seed-offset16',
           cuda_device_role='single-native-worker', capture=False)
# Row reductions use binary64 over 64*64 entries. 16*4096*u64/(1-16*4096*u64)
# is below 1e-10. The entropy allowance is the producer's fixed stopping rule
# for rounded float32 probabilities, then binary64 reduction, not a universal
# softmax accuracy theorem. Nonnegative sums/norms/losses allow signed zero only.
OBSERVATION = dict(row_fraction=dict(lower=-1e-10, upper=1+1e-10),
                   normalized_attention_entropy=dict(lower=-1e-12, upper=1+1e-6),
                   nonnegative_lower=0.0, floating_dtype='float64', logits_dtype='float32')


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def model_schema(heads):
    """Exact state and parameter order, including biases and affine LayerNorm.

    Registered branches: custom_G_type=None, reference_rope=True,
    tie_word_embeddings=False, all three bias switches=True. RoPE theta/cache
    and learned past_G_values are nonpersistent; there are no persistent
    buffers or parameter aliases in this branch. All parameters receive a
    gradient tensor under the registered full native loss (zero is allowed).
    """
    if type(heads) is not int or not 2 <= heads <= 32:
        raise ValueError('Unsupported schema head count')
    width = 64*heads
    entries = {}
    def tensor(name, shape):
        entries[name] = dict(shape=list(shape), dtype='float32', trainable=True)
    def linear(name, ni, no):
        tensor(name+'.weight', (no, ni)); tensor(name+'.bias', (no,))
    def norm(name, dim):
        tensor(name+'.weight', (dim,)); tensor(name+'.bias', (dim,))
    def glu(name, ni, hidden, no):
        linear(name+'.gluw1', ni, hidden); linear(name+'.gluw2', ni, hidden)
        linear(name+'.gluw3', hidden, no)
    tensor('decoder.embedding.weight', (VOCABULARY, width))
    for layer in range(5):
        base=f'decoder.dec_layers.{layer}'; att=base+'.mha1'
        for name in ['wq','wk','wv']: linear(att+'.'+name,width,width)
        for name in ['Wlst','blst','pwlst','alst','balst']:
            tensor(att+'.plgatt_layer.'+name,(heads,64,64))
        linear(att+'.dense',width,width)
        for unit in range(8):
            res=att+f'.reslayerAs.{unit}'
            for dense in range(2): glu(res+f'.denseAs.{dense}',64,170,64)
            norm(res+'.layernormA',64)
        norm(att+'.layernorm1',64)
        glu(base+'.ffn',width,(width*8)//3,width)
        norm(base+'.layernorm1',width); norm(base+'.layernorm2',width)
    norm('decoder.layernorm1',width)
    linear('final_layer',width,VOCABULARY)
    return entries


def optimizer_groups(heads, control):
    from model_rg.criticality import generator_parameter
    names=list(model_schema(heads))
    selected=[[n for n in names if generator_parameter(n)],
              [n for n in names if not generator_parameter(n)]]
    offset=0;groups=[]
    # Use precisely the producer's multiplication/division order.
    for group_names, rate in zip(selected,[3e-4*control,3e-4*128/(64*heads)]):
        groups.append(dict(params=list(range(offset,offset+len(group_names))),param_names=group_names,
            lr=rate,betas=(.9,.95),eps=1e-8,weight_decay=.01,amsgrad=False,maximize=False,
            foreach=False,capturable=False,differentiable=False,fused=None,decoupled_weight_decay=True))
        offset+=len(group_names)
    return groups


def runtime_check():
    observed=dict(torch=str(torch.__version__),torch_git=torch.version.git_version,
                  cuda=torch.version.cuda,transformers=importlib.metadata.version('transformers'),
                  byteorder=sys.byteorder,pointer_bits=8*struct.calcsize('P'))
    if observed != RUNTIME or torch.get_default_dtype()!=torch.float32:
        raise ValueError('Unsupported native runtime or default dtype')


def registered_binding(heads):
    runtime_check()
    return dict(schema=SCHEMA,runtime=deepcopy(RUNTIME),rng=deepcopy(RNG),
                observation=deepcopy(OBSERVATION),native_sources=dict(NATIVE_SOURCES),
                adapter_sources=dict(ADAPTER_SOURCES),
                model_schema_sha256={str(n):canonical_digest(list(model_schema(n).items())) for n in sorted(set(heads))})


def validate_binding(p, paths, repo):
    expected=registered_binding(p['design']['heads'])
    if canonical_digest(p.get('native_state_contract')) != canonical_digest(expected):
        raise ValueError('Unregistered native state schema/runtime binding')
    for name,digest in NATIVE_SOURCES.items():
        if hashlib.sha256(Path(paths[name]).read_bytes()).hexdigest()!=digest:
            raise ValueError('Unregistered native configuration/model source: '+name)
    for name,digest in ADAPTER_SOURCES.items():
        if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Unregistered native adapter source: '+name)


def tensor_contract(value, shape, label, *, dtype=torch.float32):
    if (not isinstance(value,torch.Tensor) or value.layout!=torch.strided or
        value.device.type!='cpu' or list(value.shape)!=list(shape) or value.dtype!=dtype):
        raise ValueError('Invalid tensor contract: '+label)
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite tensor: '+label)


def validate_model(model, heads):
    schema=model_schema(heads)
    if not isinstance(model,dict) or set(model)!=set(schema):
        raise ValueError('Native model keys differ from registered complete schema')
    for name,desc in schema.items(): tensor_contract(model[name],desc['shape'],'model '+name)


def validate_optimizer(optimizer, heads, control, steps):
    if not isinstance(optimizer,dict) or set(optimizer)!={'state','param_groups'}:
        raise ValueError('Invalid optimizer checkpoint schema')
    expected=optimizer_groups(heads,control);groups=optimizer['param_groups'];slots=optimizer['state']
    if not isinstance(groups,list) or len(groups)!=2 or not isinstance(slots,dict):
        raise ValueError('Invalid optimizer groups/slots')
    schema=model_schema(heads);coordinates={}
    for actual,goal in zip(groups,expected):
        if not isinstance(actual,dict) or set(actual)!=set(goal):
            raise ValueError('Invalid optimizer group metadata keys')
        for key,value in goal.items():
            x=actual[key]
            # Avoid bool/int aliases and tensor/list coercion; preserve tuple betas.
            if type(x) is not type(value) or x!=value:
                raise ValueError('Optimizer group contract differs: '+key)
        if any(type(i) is not int for i in actual['params']):
            raise ValueError('Invalid optimizer coordinate IDs')
        coordinates.update(zip(goal['params'],goal['param_names']))
    if any(type(i) is not int for i in slots) or set(slots)!=set(coordinates):
        raise ValueError('Incomplete optimizer coordinates')
    for index,name in coordinates.items():
        slot=slots[index]
        if not isinstance(slot,dict) or set(slot)!={'step','exp_avg','exp_avg_sq'}:
            raise ValueError('Missing or additional Adam slot fields')
        for field in ['exp_avg','exp_avg_sq']:
            tensor_contract(slot[field],schema[name]['shape'],'Adam '+field+' '+name)
        if bool((slot['exp_avg_sq']<0).any()):raise ValueError('Negative Adam second moment')
        tensor_contract(slot['step'],[],'Adam clock')
        if slot['step'].item()!=steps:raise ValueError('Adam clock differs from checkpoint step')


def validate_cpu_rng(state):
    tensor_contract(state,list(state.shape) if isinstance(state,torch.Tensor) else [],'CPU RNG',dtype=torch.uint8)
    if state.ndim!=1 or not state.is_contiguous():raise ValueError('Invalid CPU RNG tensor')
    first=torch.Generator(device='cpu');second=torch.Generator(device='cpu')
    try:
        first.set_state(state);second.set_state(state)
        # Legacy formats/padding do not qualify as the registered producer format.
        if state.numel()!=first.get_state().numel():raise ValueError('Noncurrent CPU RNG format')
        a=torch.randn(17,generator=first);b=torch.randn(17,generator=second)
        if not torch.equal(a,b) or not torch.equal(first.get_state(),second.get_state()):
            raise ValueError('CPU RNG continuation differs')
    except RuntimeError as exc:
        raise ValueError('Unusable CPU RNG state') from exc


def validate_cuda_rng(state):
    """CPU parser for the pinned CUDAGeneratorImpl, without a CUDA API call.

    get_state emits uint64 seed then int64 offset in native byte order. The
    setter casts the offset to uint64 and requires divisibility by four. All
    seed bits and either offset sign are legal. We require the current 16-byte
    format; the setter's legacy 8-byte seed-only format is outside this route.
    Source: PyTorch commit in RUNTIME, aten/src/ATen/cuda/CUDAGeneratorImpl.cpp,
    get_state, set_state, set_philox_offset_per_thread (see docs/ADMISSION.md).
    """
    tensor_contract(state,[16],'CUDA RNG',dtype=torch.uint8)
    if not state.is_contiguous():raise ValueError('Invalid CUDA RNG layout')
    seed,offset=struct.unpack('<QQ',state.numpy().tobytes())
    if offset%4:raise ValueError('Invalid CUDA RNG Philox offset')
    return seed,offset


def validate_checkpoint(state, heads, control, steps):
    if type(state.get('step')) is not int or state['step']!=steps:
        raise ValueError('Invalid integer checkpoint step')
    validate_model(state['model'],heads)
    validate_optimizer(state['optimizer'],heads,control,steps)
    validate_cpu_rng(state['cpu_rng']);validate_cuda_rng(state['cuda_rng'])


def validate_observations(obs, times, milestones):
    for key in obs.files:
        expected=np.dtype('int64') if key in ('steps','blocks') else np.dtype('float32' if key.startswith('logits_') else 'float64')
        if obs[key].dtype!=expected:raise ValueError('Observation dtype differs: '+key)
    fields=obs['heads']
    for index,name in enumerate(('row_fraction','normalized_attention_entropy')):
        bounds=OBSERVATION[name];a=fields[...,index]
        if np.any(a<bounds['lower']) or np.any(a>bounds['upper']):
            raise ValueError('Observation interval violation: '+name)
    if np.any(fields[...,2:]<0):raise ValueError('Negative operator RMS or row energy')
    for key in obs.files:
        if key in ('nll_path','training_loss','gradient_norm') or key.startswith('evaluation_nll_'):
            if np.any(obs[key]<0):raise ValueError('Negative observation: '+key)
    for step in milestones:
        i=int(np.flatnonzero(times==step)[0])
        if not np.array_equal(obs[f'evaluation_nll_{step}'][:64],obs['nll_path'][i]):
            raise ValueError('Overlapping NLL observations disagree')

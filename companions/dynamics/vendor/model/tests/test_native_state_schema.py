"""Independent native schema check and paired, reauthenticated semantic defects."""
from copy import deepcopy
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest
import torch

from model_rg import native_state_schema as schema
from model_rg.replay_equality import logical_equal
from test_onepass_admission import Fixture, SPEC, REPO, native, recorder, dump
from model_rg import onepass_admission as admission


@pytest.mark.parametrize('heads,device',[(2,'cpu'),(4,'cpu'),(3,'meta'),(5,'meta'),(32,'meta')])
def test_schema_against_independently_constructed_native_model(heads,device):
    from model_rg.training import TrainingModel
    from model_rg.criticality import optimizer_for
    with torch.random.fork_rng(devices=[]),torch.device(device):
        model=TrainingModel(REPO.parent/'native/PLDR-LLM-v51-SOC-110M-1',heads,73,device)
    actual=model.model.state_dict();declared=schema.model_schema(heads)
    assert list(actual)==list(declared)==[n for n,p in model.model.named_parameters()]
    assert all(list(actual[n].shape)==v['shape'] and actual[n].dtype==torch.float32 for n,v in declared.items())
    assert len({id(p) for p in model.model.parameters()})==len(declared)
    assert model.config.custom_G_type is None and model.config.reference_rope
    assert not model.config.tie_word_embeddings
    # RoPE and the runtime learned-operator cache are explicitly nonpersistent.
    assert not set(dict(model.model.named_buffers())) & set(actual)
    optimizer=optimizer_for(model,2.)
    assert optimizer.state_dict()['param_groups']==schema.optimizer_groups(heads,2.)


def test_all_registered_width_formulas():
    for heads in range(2,33):
        d=schema.model_schema(heads);w=64*heads
        assert d['decoder.embedding.weight']['shape']==[32000,w]
        assert d['decoder.dec_layers.0.ffn.gluw1.weight']['shape']==[w*8//3,w]
        assert d['decoder.dec_layers.4.mha1.plgatt_layer.Wlst']['shape']==[heads,64,64]


def guards(monkeypatch):
    counts=dict(model_construction=0,worker_dispatch=0,cuda_initialization=0,optimizer_update=0)
    def block(key):
        def fail(*a,**kw):counts[key]+=1;raise AssertionError('Forbidden early action: '+key)
        return fail
    monkeypatch.setattr(native,'TrainingModel',block('model_construction'))
    monkeypatch.setattr(native,'dispatch_worker',block('worker_dispatch'))
    monkeypatch.setattr(recorder,'dispatch_worker',block('worker_dispatch'))
    for name in ('init','_lazy_init','set_device','reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda,name,block('cuda_initialization'))
    monkeypatch.setattr(torch.optim.AdamW,'step',block('optimizer_update'))
    return counts


STATE_CASES={
    'missing_model':'model keys','additional_buffer':'model keys','string_model':'tensor contract',
    'model_shape':'tensor contract','model_dtype':'tensor contract','model_nan':'Nonfinite',
    'missing_id':'coordinates','duplicate_id':'params','extra_id':'params','wrong_group':'params',
    'coordinate_order':'param_names','id_order':'params','moment_shape':'tensor contract',
    'moment_dtype':'tensor contract','negative_second':'Negative Adam','nan_first':'Nonfinite',
    'extra_slot':'slot fields','missing_moment':'slot fields','step_rank':'tensor contract',
    'step_dtype':'tensor contract','step_value':'Adam clock','checkpoint_step':'integer checkpoint',
    'checkpoint_float':'integer checkpoint','cpu_short':'CPU RNG','cpu_field':'CPU RNG',
    'cpu_dtype':'CPU RNG','cpu_rank':'CPU RNG','cuda_short':'CUDA RNG','cuda_dtype':'CUDA RNG',
    'cuda_rank':'CUDA RNG','cuda_offset':'Philox offset','cpu_as_cuda':'CUDA RNG',
}
for field in ['lr','betas','eps','weight_decay','foreach','capturable','differentiable','fused','maximize','amsgrad','decoupled_weight_decay']:
    STATE_CASES['group_'+field]='group contract'
STATE_CASES['other_lr']='group contract'


def corrupt(state,case):
    model=state['model'];opt=state['optimizer'];g=opt['param_groups'];slots=opt['state'];first=next(iter(model));i=g[0]['params'][0];slot=slots[i]
    if case=='missing_model':del model[first]
    elif case=='additional_buffer':model['decoder.past_G_values']=torch.zeros(1)
    elif case=='string_model':model[first]='finite but not a tensor'
    elif case=='model_shape':model[first]=torch.zeros(1)
    elif case=='model_dtype':model[first]=model[first].double()
    elif case=='model_nan':model[first]=torch.tensor(float('nan')).expand(model[first].shape)
    elif case=='missing_id':del slots[i]
    elif case=='duplicate_id':g[0]['params'][1]=i
    elif case=='extra_id':g[0]['params'].append(len(model))
    elif case=='wrong_group':g[0]['params'][0],g[1]['params'][0]=g[1]['params'][0],g[0]['params'][0]
    elif case=='coordinate_order':g[0]['param_names'][0],g[0]['param_names'][1]=g[0]['param_names'][1],g[0]['param_names'][0]
    elif case=='id_order':g[0]['params'][0],g[0]['params'][1]=g[0]['params'][1],g[0]['params'][0]
    elif case=='moment_shape':slot['exp_avg']=torch.zeros(1);slot['exp_avg_sq']=torch.zeros(1)
    elif case=='moment_dtype':slot['exp_avg']=slot['exp_avg'].double()
    elif case=='negative_second':slot['exp_avg_sq']=torch.tensor(-1.).expand(slot['exp_avg_sq'].shape)
    elif case=='nan_first':slot['exp_avg']=torch.tensor(float('nan')).expand(slot['exp_avg'].shape)
    elif case=='extra_slot':slot['max_exp_avg_sq']=slot['exp_avg_sq']
    elif case=='missing_moment':del slot['exp_avg']
    elif case=='step_rank':slot['step']=slot['step'].reshape(1)
    elif case=='step_dtype':slot['step']=slot['step'].double()
    elif case=='step_value':slot['step']=torch.tensor(127.)
    elif case=='checkpoint_step':state['step']=127
    elif case=='checkpoint_float':state['step']=128.
    elif case=='cpu_short':state['cpu_rng']=state['cpu_rng'][:1]
    elif case=='cpu_field':state['cpu_rng'][8:12]=255
    elif case=='cpu_dtype':state['cpu_rng']=state['cpu_rng'].float()
    elif case=='cpu_rank':state['cpu_rng']=state['cpu_rng'][None]
    elif case=='cuda_short':state['cuda_rng']=state['cuda_rng'][:1]
    elif case=='cuda_dtype':state['cuda_rng']=state['cuda_rng'].float()
    elif case=='cuda_rank':state['cuda_rng']=state['cuda_rng'][None]
    elif case=='cuda_offset':state['cuda_rng'][8]=1
    elif case=='cpu_as_cuda':state['cuda_rng']=state['cpu_rng'].clone()
    elif case=='other_lr':g[1]['lr']=.001
    elif case.startswith('group_'):
        field=case.removeprefix('group_')
        g[0][field]={'lr':-.001,'betas':(1.1,.95),'eps':0.,'weight_decay':-.1,'foreach':True,
            'capturable':True,'differentiable':True,'fused':True,'maximize':True,'amsgrad':True,'decoupled_weight_decay':False}[field]
    else:raise AssertionError(case)


def paired_defect(f,case):
    states=[]
    for side in ['native','replay']:
        p=f.study/'qualification'/side/'final-state.pt';state=torch.load(p,weights_only=True)
        corrupt(state,case);torch.save(state,p);states.append(state)
    assert logical_equal(*states), 'Defect must agree in both copies'
    f.rehash_qualification()


@pytest.mark.parametrize('case,reason',STATE_CASES.items())
def test_paired_semantic_checkpoint_defects(tmp_path,monkeypatch,case,reason):
    f=Fixture(tmp_path);f.qualification();paired_defect(f,case);counts=guards(monkeypatch)
    with pytest.raises(ValueError,match=reason):f.check(True)
    assert all(v==0 for v in counts.values())
    assert '_admission' not in f.p


OBS_CASES={
    'negative_fraction':'row_fraction','fraction_upper':'row_fraction',
    'negative_entropy':'normalized_attention_entropy','entropy_upper':'normalized_attention_entropy',
    'negative_operator':'operator RMS','negative_energy':'row energy','negative_loss':'training_loss',
    'negative_nll':'nll_path','negative_evaluation':'evaluation_nll','negative_norm':'gradient_norm',
    'dtype':'dtype','logit_dtype':'dtype','step_dtype':'qualification steps','step_schedule':'time or source',
    'repeated_blocks':'time or source','inconsistent_nll':'Overlapping NLL',
}


@pytest.mark.parametrize('case,reason',OBS_CASES.items())
def test_paired_observation_defects(tmp_path,monkeypatch,case,reason):
    f=Fixture(tmp_path);f.qualification()
    for side in ['native','replay']:
        path=f.study/'qualification'/side/'observations.npz'
        with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        if case in ('negative_fraction','fraction_upper','negative_entropy','entropy_upper'):
            index=int('entropy' in case);a['heads'][0,0,0,0,index]=-.01 if case.startswith('negative') else 1.01
        elif case in ('negative_operator','negative_energy'):a['heads'][0,0,0,0,2 if case=='negative_operator' else 3]=-1.
        elif case in ('negative_loss','negative_nll','negative_norm','negative_evaluation'):
            key={'negative_loss':'training_loss','negative_nll':'nll_path','negative_norm':'gradient_norm','negative_evaluation':'evaluation_nll_0'}[case];a[key].flat[0]=-.01
        elif case=='dtype':a['heads']=a['heads'].astype(np.float32)
        elif case=='logit_dtype':a['logits_0']=a['logits_0'].astype(np.float64)
        elif case=='step_dtype':a['steps']=a['steps'].astype(np.float64)
        elif case=='step_schedule':a['steps'][1]=63
        elif case=='repeated_blocks':a['blocks'][1]=a['blocks'][0]
        elif case=='inconsistent_nll':a['evaluation_nll_0'][0]=1.
        np.savez(path,**a)
    f.rehash_qualification();counts=guards(monkeypatch)
    with pytest.raises(ValueError,match=reason):f.check(True)
    assert all(v==0 for v in counts.values())


@pytest.mark.parametrize('index,name',[(0,'row_fraction'),(1,'normalized_attention_entropy')])
@pytest.mark.parametrize('edge',['lower','upper'])
def test_interval_boundaries(tmp_path,index,name,edge):
    f=Fixture(tmp_path);f.qualification();path=f.study/'qualification/native/observations.npz'
    with np.load(path) as z:a={k:z[k] for k in z.files}
    bound=schema.OBSERVATION[name][edge]
    class Observations(dict):
        @property
        def files(self):return list(self)
    a=Observations(a);a['heads'][...,index]=bound
    schema.validate_observations(a,a['steps'],[0,128])
    a['heads'][...,index]=np.nextafter(bound,-np.inf if edge=='lower' else np.inf)
    with pytest.raises(ValueError,match=name):schema.validate_observations(a,a['steps'],[0,128])


@pytest.mark.parametrize('preparation,observer',[('base','native'),('refinement','native'),('clock','native'),('refinement','shared'),('clock','shared')])
def test_complete_acceptance_keeps_early_action_guards_zero(tmp_path,monkeypatch,preparation,observer):
    f=Fixture(tmp_path,preparation,observer);f.qualification();counts=guards(monkeypatch)
    # Legitimate signed weights/moments/logits and norms above the clip limit.
    for side in ['native','replay']:
        path=f.study/'qualification'/side/'observations.npz'
        with np.load(path) as z:a={k:z[k] for k in z.files}
        a['gradient_norm'][:]=17.;a['heads'][...,2:]=2.;np.savez(path,**a)
    f.rehash_qualification();before=torch.get_rng_state().clone()
    assert f.check(True)['_admission']['qualification_evidence']=='passed'
    assert torch.equal(torch.get_rng_state(),before)
    assert all(v==0 for v in counts.values())


@pytest.mark.parametrize('field',['schema','runtime','rng','model_schema_sha256'])
def test_unknown_binding_rejected(tmp_path,field):
    f=Fixture(tmp_path);f.p['native_state_contract'][field]='unregistered'
    with pytest.raises(ValueError,match='binding'):f.check()


@pytest.mark.parametrize('entry',['parent','direct_worker','shared_parent','shared_worker'])
def test_semantic_failure_at_native_boundaries(tmp_path,monkeypatch,entry):
    f=Fixture(tmp_path,'refinement','shared');f.qualification();paired_defect(f,'moment_shape')
    # Inject only miniature external assets; exercise the actual adapter and
    # parent/worker sequence, preserving all schema/runtime/semantic checks.
    original=admission.validate_protocol
    def helper(p,study,repo,locations,**kwargs):
        return original(p,study,repo,f.paths,spec=SPEC,**kwargs)
    monkeypatch.setattr(native,'validate_protocol',helper);counts=guards(monkeypatch)
    job=f.p['jobs'][0];dest=f.study/'runs'/job['run_id']
    calls={'parent':lambda:native.run(f.study),'direct_worker':lambda:native.train(f.study,job,'cuda:0','scientific',dest),
        'shared_parent':lambda:recorder.run(f.study),'shared_worker':lambda:recorder.train(f.study,job,'cuda:0','scientific',dest)}
    with pytest.raises(ValueError,match='tensor contract'):calls[entry]()
    assert not dest.exists() and all(v==0 for v in counts.values())


def test_cpu_rng_state_is_local_and_usable():
    before=torch.get_rng_state().clone();state=torch.Generator().manual_seed(37).get_state()
    schema.validate_cpu_rng(state)
    assert torch.equal(before,torch.get_rng_state())


@pytest.mark.parametrize('seed,offset',[(0,0),(2**64-1,4),(17,2**63),(3,2**64-4)])
def test_cuda_parser_valid_bit_patterns(seed,offset):
    state=torch.tensor(list(struct.pack('<QQ',seed,offset)),dtype=torch.uint8)
    assert schema.validate_cuda_rng(state)==(seed,offset)

@pytest.mark.parametrize('role',['native_model','native_config'])
def test_rehash_does_not_register_changed_native_source(tmp_path,role):
    from model_rg.provenance import sha256
    f=Fixture(tmp_path);path=f.paths[role]
    path.write_text(path.read_text()+'\n# Unregistered source change.\n')
    f.p['input_sha256'][str(path)]=sha256(path)
    with pytest.raises(ValueError,match='Unregistered native'):f.check()


@pytest.mark.parametrize('historical_contract', [False, True])
def test_preserved_native_source_cannot_qualify_as_current(tmp_path, historical_contract):
    from model_rg.provenance import sha256
    root = Path(__file__).resolve().parents[5]
    adaptation = json.loads((root / 'provenance/native-comment-adaptation.json').read_text())
    historical = root / adaptation['historical_source']
    assert sha256(historical) == adaptation['historical_source_sha256']
    f = Fixture(tmp_path)
    f.paths['native_model'].write_bytes(historical.read_bytes())
    f.p['input_sha256'][str(f.paths['native_model'])] = sha256(historical)
    if historical_contract:
        f.p['native_state_contract']['native_sources']['native_model'] = sha256(historical)
    with pytest.raises(ValueError, match='Unregistered native'):
        f.check()


@pytest.mark.parametrize('device',['cpu','cuda','cuda:-1',None])
def test_qualification_rng_device_role(tmp_path,device):
    f=Fixture(tmp_path);f.qualification()
    for side in ['native','replay']:
        path=f.study/'qualification'/side/'manifest.json';m=json.loads(path.read_text());m['device']=device;dump(path,m)
    f.rehash_qualification()
    with pytest.raises(ValueError,match='device role'):f.check(True)


def test_unsupported_local_runtime_rejects_before_cuda(tmp_path,monkeypatch):
    f=Fixture(tmp_path);counts=guards(monkeypatch);monkeypatch.setattr(torch.version,'git_version','unknown')
    with pytest.raises(ValueError,match='runtime'):f.check()
    assert all(v==0 for v in counts.values())


@pytest.mark.parametrize('field,value',[('capture',0),('pointer_bits',64.)])
def test_binding_preserves_json_types(tmp_path,field,value):
    f=Fixture(tmp_path);sector='rng' if field=='capture' else 'runtime'
    f.p['native_state_contract'][sector][field]=value
    with pytest.raises(ValueError,match='binding'):f.check()


def test_failed_readmission_clears_transient_success(tmp_path):
    f=Fixture(tmp_path);f.qualification();p=f.check(True)
    assert p['_admission']['qualification_evidence']=='passed'
    p['source_sha256']={}
    with pytest.raises(ValueError):
        admission.validate_protocol(p,f.study,REPO,f.paths,scientific=True,spec=SPEC)
    assert '_admission' not in p and '_validated_vocabulary' not in p

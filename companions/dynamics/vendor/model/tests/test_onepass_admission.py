"""Complete CPU fixtures for prospective admission; no native qualification runs.

The miniature assets exercise the helper contract only. Production-adapter
checks separately prove the fixed production specification and entry ordering.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import shutil
import struct
import zipfile

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_critical_onepass as native
import run_critical_shared_paths as recorder
from model_rg import onepass_admission as admission
from model_rg import native_state_schema as schema
from model_rg.provenance import sha256
from model_rg.replay_equality import contract

SPEC=admission.AssetSpec(documents=256,columns=17,probe_documents=16,probe_start=4,
                         probe_count=4,batch=2,prefix=2,segments=8)
REPO=Path(__file__).resolve().parents[1]


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


class Fixture:
    def __init__(self, root, preparation='base', observer='native'):
        self.study=root/'study';self.study.mkdir()
        corpus=root/'corpus';probes=root/'probes';model=root/'native'
        for folder in (corpus,probes,model):folder.mkdir()
        self.paths=admission.input_locations(corpus,probes,model)
        d=dict(name='fixture',heads=[2],controls=[0.,2.],seeds=[11,12,13],environments=[0,1],steps=256,shared_seed=7)
        self.p=dict(schema='critical-onepass-v1',role='scientific',admission_contract=admission.CONTRACT,
            variant=dict(preparation=preparation,observer=observer),design=d,replay_contract=contract(),
            qualification=dict(steps=128,heads=2,control=2.,full_state_replay=True,role='qualification'),
            source=dict(master_documents=SPEC.documents,population_documents=128,population_blocks=1024,
                batch=2,prefix_tokens=2,target_tokens=1,no_repeated_supervised_positions=True,
                partition_seed=9152501,stream_seed='9152502+environment'),
            observations=dict(cadence=64,head_contexts=64,evaluation_contexts=128,full_vocabulary_contexts=8,
                row_denominator_floor=1e-30,milestone_steps=[0,256]),
            architecture=dict(depth=5,head_dimension=64,generator_width=170,generator_residual_units=8),
            optimizer=dict(generator_rate='0.0003*g',other_rate='0.0003*2/N',betas=[.9,.95],epsilon=1e-8,
                weight_decay=.01,global_norm_clip=1.,foreach=False,schedule='constant'),
            arithmetic=dict(dtype='float32',tf32=False,observation_reductions='float64',cpu_threads=2),memory_ceiling_bytes=22*1024**3)
        np.save(corpus/'tokens.npy',np.arange(256*17,dtype=np.int32).reshape(256,17)%7)
        np.save(probes/'tokens.npy',np.arange(16*17,dtype=np.int32).reshape(16,17)%7)
        np.save(probes/'offsets.npy',np.arange(16,dtype=np.int64)%14)
        for prefix,count in [('corpus',256),('probe',16)]:
            dump(self.paths[prefix+'_records'],[dict(content_sha256=hashlib.sha256(f'{prefix}-{i}'.encode()).hexdigest()) for i in range(count)])
        dump(corpus/'manifest.json',dict(schema='onepass-refinedweb-corpus-v1',status='complete',training_documents=256,
            shape=[256,17],dtype='int32',tokens_sha256=sha256(corpus/'tokens.npy'),records_sha256=sha256(corpus/'records.json')))
        dump(probes/'manifest.json',dict(schema='controlled-cohort-v1',kind='short',count=16,length=17,
            tokens_sha256=sha256(probes/'tokens.npy'),offsets_sha256=sha256(probes/'offsets.npy'),records_sha256=sha256(probes/'records.json')))
        for name in ['modeling_pldrllm.py','configuration_pldrllm.py']:
            shutil.copyfile(REPO.parent/'native/PLDR-LLM-v51-SOC-110M-1'/name,model/name)
        self.p['native_state_contract']=schema.registered_binding(d['heads'])
        extra={}
        if preparation!='base':
            search=root/'search.json';dump(search,dict(status='complete',design=dict(seeds=[1,2])))
            decision=dict(search_analyses={str(search):sha256(search)},hypotheses=['finite peak'],primary_fields=['row'],held_out_axes=['seed'],decision_rules=['retain failures'])
            self.p['selection']=decision;path=self.study/'selection-decision.json';dump(path,decision)
            self.p['selection_record_sha256']=sha256(path);extra.update(selection_decision=path,search_analysis_0=search)
        if preparation=='clock':
            f=dict(name='fixture',heads=[2,4],rate_ratios=[0.,2.],seeds=d['seeds'],environments=d['environments'],steps_per_head=128,corpus_documents_per_head=32,shared_seed=7)
            d['name']='fixture-h2';self.p['source'].update(population_documents=64,population_blocks=512,population_policy='nested-width-prefix-v1')
            path=self.study/'family.json';dump(path,f);extra['family_specification']=path
            self.p['joint_family']=dict(specification=f,specification_sha256=sha256(path),terminal_other_clock=.0006*128,terminal_generator_clocks=[.0006*128*r for r in f['rate_ratios']],consumed_fraction=1.)
            self.p['observations']['milestone_steps']=[0,64,128,256]
        if observer=='shared':
            self.p['shared_paths']=dict(schema='native-shared-parameter-path-v1',parameter_selector='reslayerAs',cadence=256,include_first_update=True,first_raw_and_clipped_gradient=True)
            self.p['qualification']['uninstrumented_comparison']=True
        self.p['jobs']=[dict(run_id=f'e{e}-h{h}-g{g:g}-s{s}',environment=e,heads=h,control=g,seed=s,steps=256,shared_seed=7,save_optimizer=preparation!='base' or s in d['seeds'][:2]) for e in d['environments'] for g in d['controls'] for h in d['heads'] for s in d['seeds']]
        self.p['source_sha256']={name:sha256(REPO/name) for name in admission.registered_sources(self.p)}
        self.p['input_roles']={k:str(v) for k,v in {**self.paths,**extra}.items()}
        self.p['input_sha256']={str(v):sha256(v) for v in {**self.paths,**extra}.values()}
        dump(self.study/'design.json',d);self.p['design_sha256']=sha256(self.study/'design.json')
        pr=np.arange(4,8);pt=np.load(probes/'tokens.npy');po=np.load(probes/'offsets.npy')
        self.arrays=dict(probe_rows=pr,probes=pt[pr[:,None],po[pr,None]+np.arange(3)])
        halves=np.random.default_rng(9152501).permutation(256).reshape(2,-1)
        for e in d['environments']:
            pop=halves[e,:self.p['source']['population_documents']]
            local=np.random.default_rng(9152502+e).permutation(len(pop)*8)[:512]
            self.arrays[f'population_{e}']=pop;self.arrays[f'blocks_{e}']=(8*pop[local//8]+local%8).reshape(256,2)
        self.save_arrays()

    def save_arrays(self):
        np.savez(self.study/'selection.npz',**self.arrays)
        self.p['selection_sha256']=sha256(self.study/'selection.npz');self.save()

    def save(self):dump(self.study/'protocol.json',self.p)

    def check(self,scientific=False):
        self.save();native.validate_design(self.p['design'])
        return admission.validate_protocol(deepcopy(self.p),self.study,REPO,self.paths,scientific=scientific,spec=SPEC)

    def qualification(self):
        self.save();ph=sha256(self.study/'protocol.json');job=admission.qualification_job(self.p)
        # Full native dimensions and every parameter, with explicitly synthetic
        # values. Expanded scalar storage keeps the fixture small on disk;
        # tensor strides do not change the checkpoint's logical values.
        desc=schema.model_schema(job['heads'])
        model={n:torch.tensor(-.125).expand(v['shape']) for n,v in desc.items()}
        groups=schema.optimizer_groups(job['heads'],job['control']);slots={}
        for g in groups:
            for i,n in zip(g['params'],g['param_names']):
                slots[i]=dict(step=torch.tensor(128.),exp_avg=torch.tensor(-.25).expand(desc[n]['shape']),exp_avg_sq=torch.tensor(.5).expand(desc[n]['shape']))
        state=dict(model=model,optimizer=dict(state=slots,param_groups=groups),step=128,job=job,
            protocol_sha256=ph,cpu_rng=torch.Generator(device='cpu').manual_seed(7).get_state(),
            cuda_rng=torch.tensor(list(struct.pack('<QQ',7,0)),dtype=torch.uint8))
        self.q=dict(status='passed',protocol_sha256=ph,replay_contract=contract(),full_state_bitwise_replay=True,scientific_updates=0,qualification_updates=256,checked_sha256={})
        for side in ['native','replay']:
            path=self.study/'qualification'/side;path.mkdir(parents=True)
            torch.save(state,path/'final-state.pt')
            obs=dict(steps=np.array([0,64,128]),heads=np.zeros((3,64,5,2,4)),nll_path=np.zeros((3,64)),blocks=self.arrays['blocks_0'][:128],training_loss=np.ones(128),gradient_norm=np.ones(128))
            for t in sorted(({0,128}|set(self.p['observations']['milestone_steps']))&{0,64,128}):
                obs[f'logits_{t}']=np.full((8,schema.VOCABULARY),-.25,dtype=np.float32);obs[f'evaluation_nll_{t}']=np.zeros(128)
            np.savez(path/'observations.npz',**obs)
            m=dict(status='complete',role='qualification',job=job,protocol_sha256=ph,producer_sha256=self.p['source_sha256'],completed_steps=128,attempted_steps=128,intended_steps=128,scientific_updates=0,qualification_updates=128,optimizer_saved=True,checkpoint_available=True,checkpoint_resumable=True,optimizer_partial_mutation=False,original_failure=None,artifacts={},device='cuda:0',native_state_contract=self.p['native_state_contract'])
            if side=='replay' and self.p['variant']['observer']=='shared':
                names=[n for n in model if 'reslayerAs' in n]
                endpoint=np.concatenate([model[n].reshape(-1).numpy() for n in names])
                np.save(path/'shared-parameters.npy',np.stack([endpoint,endpoint,endpoint]));np.save(path/'shared-steps.npy',np.array([0,1,128]))
                np.savez(path/'first-gradient.npz',raw_gradient=np.ones_like(endpoint),clipped_gradient=np.ones_like(endpoint))
                dump(path/'shared-metadata.json',dict(schema='native-shared-parameter-path-v1',protocol_sha256=ph,role='qualification',job=job,names=names,shapes=[list(model[n].shape) for n in names],steps=[0,1,128],shared_parameter_count=len(endpoint)))
                self.q.update(uninstrumented_vs_instrumented_bitwise=True,shared_endpoint_matches_state=True)
            dump(path/'manifest.json',m)
        self.rehash_qualification()

    def rehash_qualification(self):
        checked={}
        for side in ['native','replay']:
            path=self.study/'qualification'/side;m=json.loads((path/'manifest.json').read_text())
            names=['final-state.pt','observations.npz']+ (list(admission.RECORDER_ARTIFACTS) if side=='replay' and self.p['variant']['observer']=='shared' else [])
            m['artifacts']={name:sha256(path/name) for name in names};dump(path/'manifest.json',m)
            checked.update({str(path/name):sha256(path/name) for name in ['manifest.json',*names]})
        self.q['checked_sha256']=checked;dump(self.study/'qualification/verification.json',self.q)


@pytest.fixture
def fixture(tmp_path):return Fixture(tmp_path)


@pytest.mark.parametrize('preparation,observer',[('base','native'),('refinement','native'),('clock','native'),('refinement','shared'),('clock','shared')])
@pytest.mark.parametrize('scientific',[False,True])
def test_valid_registered_contracts(tmp_path,preparation,observer,scientific):
    f=Fixture(tmp_path,preparation,observer)
    if scientific:f.qualification()
    result=f.check(scientific)['_admission']
    assert result==dict(structural_preflight='passed',qualification_evidence='passed' if scientific else 'not_requested',completed_scientific_acquisition=False,native_updates_executed=0,asset_contract='helper_fixture')


@pytest.mark.parametrize('name',admission.CORE_SOURCES)
def test_each_missing_source(fixture,name):
    del fixture.p['source_sha256'][name]
    with pytest.raises(ValueError,match='source roles'):fixture.check()


@pytest.mark.parametrize('role',admission.BASE_ROLES)
def test_each_missing_input(fixture,role):
    del fixture.p['input_sha256'][fixture.p['input_roles'][role]]
    with pytest.raises(ValueError,match='Input digest roles'):fixture.check()


@pytest.mark.parametrize('scientific',[False,True])
@pytest.mark.parametrize('missing',['input_sha256','source_sha256','training'])
def test_six_reported_incomplete_preflights(fixture,scientific,missing,monkeypatch):
    if missing=='training':del fixture.p['source_sha256']['src/model_rg/training.py']
    else:fixture.p[missing]={}
    (fixture.study/'selection.npz').write_bytes(b'not an npz archive')
    fixture.p['selection_sha256']=sha256(fixture.study/'selection.npz');fixture.save()
    monkeypatch.setattr(native,'REPO',REPO)
    monkeypatch.setattr(native,'TrainingModel',lambda *a,**k:pytest.fail('model construction'))
    monkeypatch.setattr(native,'dispatch_worker',lambda *a,**k:pytest.fail('worker dispatch'))
    monkeypatch.setattr(torch.cuda,'init',lambda *a:pytest.fail('CUDA initialization'))
    with pytest.raises(ValueError,match='source|input|Input'):native.admit(fixture.study,scientific)


@pytest.mark.parametrize('case',['empty_source','empty_input','bad_digest','extra_source','wrong_variant','extra_input','ambiguous_input','wrong_role','bad_map_type','bad_digest_type','role_missing'])
def test_roles_reject_individual_defects(fixture,case):
    p=fixture.p
    if case=='empty_source':p['source_sha256']={}
    elif case=='empty_input':p['input_sha256']={}
    elif case=='bad_digest':p['source_sha256'][admission.CORE_SOURCES[0]]='0'*64
    elif case=='extra_source':p['source_sha256']['unregistered.py']='0'*64
    elif case=='wrong_variant':p['variant']['preparation']='unknown'
    elif case=='extra_input':p['input_sha256'][str(fixture.study/'foreign')]='0'*64
    elif case=='ambiguous_input':p['input_roles']['probe_tokens']=p['input_roles']['corpus_tokens']
    elif case=='wrong_role':p['input_roles']['native_model']=str(fixture.study/'modeling_pldrllm.py')
    elif case=='bad_map_type':p['source_sha256']=[]
    elif case=='bad_digest_type':p['input_sha256'][next(iter(p['input_sha256']))]=7
    elif case=='role_missing':del p['input_roles']['corpus_tokens']
    with pytest.raises(ValueError):fixture.check()


@pytest.mark.parametrize('case',['missing_key','extra_key','bad_shape','float_indices','object','negative','repeated','outside_population','wrong_population','reversed_blocks','probe_rows','probe_tokens','nonfinite','truncated','not_archive'])
def test_selection_rejects_individual_defects(fixture,case):
    a=fixture.arrays
    if case=='missing_key':del a['blocks_0']
    elif case=='extra_key':a['unused']=np.zeros(1)
    elif case=='bad_shape':a['blocks_0']=a['blocks_0'][:-1]
    elif case=='float_indices':a['blocks_0']=a['blocks_0'].astype(float)
    elif case=='object':a['blocks_0']=a['blocks_0'].astype(object)
    elif case=='negative':a['blocks_0'][0,0]=-1
    elif case=='repeated':a['blocks_0'][0]=a['blocks_0'][1]
    elif case=='outside_population':a['blocks_0'][0,0]=99999
    elif case=='wrong_population':a['population_0']=a['population_1'].copy()
    elif case=='reversed_blocks':a['blocks_0']=a['blocks_0'][::-1]
    elif case=='probe_rows':a['probe_rows'][0]=100
    elif case=='probe_tokens':a['probes'][0,0]+=1
    elif case=='nonfinite':a['blocks_0']=a['blocks_0'].astype(float);a['blocks_0'][0,0]=np.nan
    fixture.save_arrays()
    if case in ('truncated','not_archive'):
        path=fixture.study/'selection.npz';path.write_bytes(path.read_bytes()[:20] if case=='truncated' else b'not npz')
        fixture.p['selection_sha256']=sha256(path)
    with pytest.raises((ValueError,zipfile.BadZipFile)):fixture.check()


@pytest.mark.parametrize('case',['missing','duplicate','wrong_id','wrong_steps','wrong_seed','wrong_shared_seed','checkpoint_policy','non_bool_policy'])
def test_frozen_job_contract(fixture,case):
    jobs=fixture.p['jobs']
    if case=='missing':jobs.pop()
    elif case=='duplicate':jobs[-1]=deepcopy(jobs[0])
    else:
        field,value={'wrong_id':('run_id','../escape'),'wrong_steps':('steps',128),'wrong_seed':('seed',99),'wrong_shared_seed':('shared_seed',99),'checkpoint_policy':('save_optimizer',False),'non_bool_policy':('save_optimizer',1)}[case]
        jobs[0][field]=value
    with pytest.raises(ValueError,match='job'):fixture.check()


@pytest.mark.parametrize('side',['native','replay'])
@pytest.mark.parametrize('name',admission.BASE_ARTIFACTS)
def test_each_missing_qualification_artifact(fixture,side,name):
    fixture.qualification();del fixture.q['checked_sha256'][str(fixture.study/'qualification'/side/name)]
    dump(fixture.study/'qualification/verification.json',fixture.q)
    with pytest.raises(ValueError,match='artifact roles'):fixture.check(True)


@pytest.mark.parametrize('case',['empty','protocol','comparator','claim','accounting','stale','manifest_status','manifest_steps','manifest_optimizer','state_mismatch','state_nan','missing_moment','observation_mismatch','observation_nan','observation_missing'])
def test_qualification_content_even_after_rehash(fixture,case):
    fixture.qualification();q=fixture.q;path=fixture.study/'qualification/replay'
    if case=='empty':q['checked_sha256']={}
    elif case=='protocol':q['protocol_sha256']='0'*64
    elif case=='comparator':q['replay_contract']['comparator_sha256']='0'*64
    elif case=='claim':q['full_state_bitwise_replay']=False
    elif case=='accounting':q['scientific_updates']=1
    elif case=='stale':q['checked_sha256'][str(path/'final-state.pt')]='0'*64
    elif case.startswith('manifest'):
        m=json.loads((path/'manifest.json').read_text())
        field,value={'manifest_status':('status','failed'),'manifest_steps':('completed_steps',127),'manifest_optimizer':('optimizer_saved',False)}[case]
        m[field]=value;dump(path/'manifest.json',m);fixture.rehash_qualification()
    elif case in ('state_mismatch','state_nan','missing_moment'):
        sides=['native','replay'] if case in ('state_nan','missing_moment') else ['replay']
        for side in sides:
            target=fixture.study/'qualification'/side/'final-state.pt';state=torch.load(target,weights_only=True)
            if case=='missing_moment':del state['optimizer']['state'][0]['exp_avg']
            else:
                key=next(iter(state['model']));state['model'][key]=state['model'][key].clone()
                state['model'][key].view(-1)[0]=float('nan') if case=='state_nan' else 8.
            torch.save(state,target)
        fixture.rehash_qualification()
    else:
        sides=['native','replay'] if case=='observation_nan' else ['replay']
        for side in sides:
            target=fixture.study/'qualification'/side/'observations.npz'
            with np.load(target) as saved:a={k:saved[k] for k in saved.files}
            if case=='observation_missing':del a['heads']
            else:a['heads'][0,0,0,0,0]=np.nan if case=='observation_nan' else .2
            np.savez(target,**a)
        fixture.rehash_qualification()
    dump(fixture.study/'qualification/verification.json',q)
    with pytest.raises(ValueError):fixture.check(True)


@pytest.mark.parametrize('name',admission.RECORDER_ARTIFACTS)
def test_shared_extra_roles_required(tmp_path,name):
    f=Fixture(tmp_path,'refinement','shared');f.qualification()
    del f.q['checked_sha256'][str(f.study/'qualification/replay'/name)];dump(f.study/'qualification/verification.json',f.q)
    with pytest.raises(ValueError,match='artifact roles'):f.check(True)


def test_shared_endpoint_recomputed(tmp_path):
    f=Fixture(tmp_path,'refinement','shared');f.qualification();path=f.study/'qualification/replay/shared-parameters.npy'
    a=np.load(path);a[-1,0]+=1;np.save(path,a);f.rehash_qualification()
    with pytest.raises(ValueError,match='replay mismatch'):f.check(True)


def test_qualification_job_is_separate(fixture):
    p=fixture.p;job=admission.qualification_job(p)
    assert job not in p['jobs'];admission.validate_job(p,job,'qualification')
    with pytest.raises(ValueError):admission.validate_job(p,job,'scientific')
    with pytest.raises(ValueError):admission.validate_job(p,p['jobs'][0],'qualification')


def test_production_adapter_has_no_miniature_option(fixture,monkeypatch):
    seen=[]
    def check(p,study,repo,locations,*,scientific=False,spec=admission.PRODUCTION,resolve=Path):
        seen.append(spec);assert spec==admission.PRODUCTION;return p
    monkeypatch.setattr(native,'validate_protocol',check)
    native.admit(fixture.study)
    assert seen==[admission.AssetSpec()]
    with pytest.raises(TypeError):native.admit(fixture.study,spec=SPEC)


def test_production_dimensions_reject_small_complete_assets(fixture):
    with pytest.raises(ValueError,match='dimensions'):
        admission.validate_protocol(deepcopy(fixture.p),fixture.study,REPO,fixture.paths)


@pytest.mark.parametrize('entry',['parent','direct_worker','qualify','shared_parent','shared_worker'])
def test_native_boundaries_reject_before_expensive_work(fixture,monkeypatch,entry):
    fixture.p['source_sha256']={};fixture.save()
    def forbidden(*a,**k):pytest.fail('expensive operation occurred before admission')
    monkeypatch.setattr(native,'dispatch_worker',forbidden);monkeypatch.setattr(recorder,'dispatch_worker',forbidden)
    monkeypatch.setattr(native,'TrainingModel',forbidden)
    for name in ['init','set_device','reset_peak_memory_stats']:monkeypatch.setattr(torch.cuda,name,forbidden)
    job=fixture.p['jobs'][0];dest=fixture.study/'runs'/job['run_id']
    calls={'parent':lambda:native.run(fixture.study),'direct_worker':lambda:native.train(fixture.study,job,'cuda:0','scientific',dest),
           'qualify':lambda:native.qualify(fixture.study,'cuda:0'),'shared_parent':lambda:recorder.run(fixture.study),
           'shared_worker':lambda:recorder.train(fixture.study,job,'cuda:0','scientific',dest)}
    with pytest.raises(ValueError,match='source roles'):calls[entry]()
    assert not dest.exists()

@pytest.mark.parametrize('case',['corpus_manifest','probe_manifest','duplicate_document','evaluation_overlap','token_bound','offset_bound','model_shape','probe_shape'])
def test_authenticated_input_content_rejects_after_rehash(fixture,case):
    paths=fixture.paths
    if case in ('corpus_manifest','probe_manifest'):
        role=case;m=json.loads(paths[role].read_text());m['tokens_sha256']='0'*64;dump(paths[role],m)
    elif case in ('duplicate_document','evaluation_overlap'):
        role='corpus_records' if case=='duplicate_document' else 'probe_records';rows=json.loads(paths[role].read_text())
        rows[1 if case=='duplicate_document' else 4]=rows[0] if case=='duplicate_document' else json.loads(paths['corpus_records'].read_text())[0]
        dump(paths[role],rows)
        mp=paths['corpus_manifest' if case=='duplicate_document' else 'probe_manifest'];m=json.loads(mp.read_text());m['records_sha256']=sha256(paths[role]);dump(mp,m)
    else:
        role={'token_bound':'corpus_tokens','offset_bound':'probe_offsets','model_shape':'corpus_tokens','probe_shape':'probe_tokens'}[case]
        a=np.load(paths[role])
        if case=='token_bound':a[0,0]=schema.VOCABULARY
        elif case=='offset_bound':a[0]=999
        else:a=a[:-1]
        np.save(paths[role],a)
        prefix='corpus' if role.startswith('corpus') else 'probe';m=json.loads(paths[prefix+'_manifest'].read_text());m[role.split('_')[1]+'_sha256']=sha256(paths[role]);dump(paths[prefix+'_manifest'],m)
    fixture.p['input_sha256']={path:sha256(path) for path in fixture.p['input_sha256']}
    with pytest.raises(ValueError):fixture.check()


@pytest.mark.parametrize('case',['unknown_population','source_seed','exhaustion','family_horizon','family_resource','family_digest','selection_digest','selection_seed','missing_variant_source','shared_claim'])
def test_variant_specific_contract_drift(tmp_path,case):
    f=Fixture(tmp_path,'clock','shared' if case=='shared_claim' else 'native');p=f.p
    if case=='unknown_population':p['source']['population_policy']='unknown'
    elif case=='source_seed':p['source']['partition_seed']=1
    elif case=='exhaustion':p['source']['population_documents']=1
    elif case=='family_horizon':p['joint_family']['specification']['steps_per_head']=256
    elif case=='family_resource':p['joint_family']['consumed_fraction']=.1
    elif case=='family_digest':p['joint_family']['specification_sha256']='0'*64
    elif case=='selection_digest':p['selection_record_sha256']='0'*64
    elif case=='selection_seed':p['selection']['search_analyses']={}
    elif case=='missing_variant_source':del p['source_sha256']['scripts/prepare_critical_clock_family.py']
    elif case=='shared_claim':p['shared_paths']['include_first_update']=False
    with pytest.raises(ValueError):f.check()


def test_production_resolver_accepts_full_locations_and_declared_aliases(tmp_path,monkeypatch):
    assert native.resolve_input_path(str(tmp_path/'tokens.npy')) == tmp_path/'tokens.npy'
    monkeypatch.setenv('MODEL_RG_DATA_ROOT',str(tmp_path))
    assert native.resolve_input_path('/pldr-data/model/tokens.npy') == tmp_path/'tokens.npy'
    with pytest.raises(ValueError,match='absolute'):native.resolve_input_path('tokens.npy')
    with pytest.raises(ValueError,match='Unmapped'):native.resolve_input_path('/pldr-unknown/tokens.npy')

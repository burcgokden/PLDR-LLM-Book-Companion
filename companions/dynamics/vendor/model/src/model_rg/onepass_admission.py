"""CPU admission for the explicitly registered prospective single-pass routes.

Historical protocols retain their executed sources. This contract validates
complete roles and recorded qualification, without performing native updates.
The miniature AssetSpec is a helper-test facility; the production adapter never
accepts it as an acquisition specification.
"""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import ast
import json
import re
import zipfile

import numpy as np
import torch
from model_rg.provenance import sha256
from model_rg.replay_equality import contract, require_replay, require_observations, finite_state

from model_rg.native_state_schema import validate_binding, validate_checkpoint, validate_observations, canonical_digest

CONTRACT = 'critical-onepass-admission-v2'
CORE_SOURCES = (
    'scripts/run_critical_onepass.py', 'src/model_rg/criticality.py',
    'src/model_rg/training.py', 'src/model_rg/native.py',
    'src/model_rg/variance_family.py', 'src/model_rg/controlled.py',
    'src/model_rg/provenance.py', 'src/model_rg/replay_equality.py',
    'src/model_rg/run_outcome.py', 'src/model_rg/onepass_admission.py',
    'src/model_rg/native_state_schema.py')
BASE_ROLES = ('corpus_tokens', 'corpus_records', 'corpus_manifest',
              'probe_tokens', 'probe_offsets', 'probe_records', 'probe_manifest',
              'native_model', 'native_config')
BASE_ARTIFACTS = ('manifest.json', 'final-state.pt', 'observations.npz')
RECORDER_ARTIFACTS = ('shared-parameters.npy', 'shared-steps.npy',
                      'first-gradient.npz', 'shared-metadata.json')


@dataclass(frozen=True)
class AssetSpec:
    documents: int = 524288
    columns: int = 513
    probe_documents: int = 2048
    probe_start: int = 1024
    probe_count: int = 128
    batch: int = 32
    prefix: int = 64
    segments: int = 8


PRODUCTION = AssetSpec()


def read_json(path):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('Duplicate JSON field: '+k)
            out[k] = v
        return out
    def bad(value):
        raise ValueError('Nonfinite JSON number: '+value)
    try:
        return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=bad)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Unreadable JSON artifact: '+str(path)) from exc


def digest_map(value, label):
    if not isinstance(value, dict) or not value:
        raise ValueError('Missing '+label+' roles')
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or
           re.fullmatch('[0-9a-f]{64}', v) is None for k, v in value.items()):
        raise ValueError('Malformed '+label+' digest map')
    return value


def exact_keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        found = set(value) if isinstance(value, dict) else set()
        raise ValueError(label+' roles differ; missing='+str(sorted(set(expected)-found))+
                         '; extra='+str(sorted(found-set(expected))))


def file_hash(path, digest, label):
    if not Path(path).is_file() or sha256(path) != digest:
        raise ValueError('Changed or missing '+label+': '+str(path))


def registered_sources(p):
    if p.get('admission_contract') != CONTRACT:
        raise ValueError('Missing prospective admission contract')
    v = p.get('variant')
    exact_keys(v, ['preparation', 'observer'], 'variant')
    if v['preparation'] not in ('base', 'refinement', 'clock') or v['observer'] not in ('native', 'shared'):
        raise ValueError('Unknown registered variant')
    if v['observer'] == 'shared' and v['preparation'] == 'base':
        raise ValueError('Shared paths require a refinement selection')
    names = set(CORE_SOURCES)
    if v['preparation'] in ('refinement', 'clock'):
        names.add('scripts/prepare_critical_refinement.py')
    if v['preparation'] == 'clock':
        names.add('scripts/prepare_critical_clock_family.py')
    if v['observer'] == 'shared':
        names.add('scripts/run_critical_shared_paths.py')
    for key, expected in [('selection', v['preparation'] != 'base'),
                          ('joint_family', v['preparation'] == 'clock'),
                          ('shared_paths', v['observer'] == 'shared')]:
        if (key in p) != expected:
            raise ValueError('Variant disagrees with '+key)
    return names


def input_locations(corpus, probes, native):
    paths = [corpus/'tokens.npy', corpus/'records.json', corpus/'manifest.json',
             probes/'tokens.npy', probes/'offsets.npy', probes/'records.json', probes/'manifest.json',
             native/'modeling_pldrllm.py', native/'configuration_pldrllm.py']
    return dict(zip(BASE_ROLES, paths))


def qualification_job(p):
    q = p['qualification']
    return dict(p['jobs'][0], run_id='widest-qualification', heads=q['heads'],
                control=q['control'], steps=q['steps'], save_optimizer=True)


def validate_job(p, job, role):
    if role == 'scientific':
        if job not in p['jobs']:
            raise ValueError('Unlisted scientific job')
    elif role == 'qualification':
        if job != qualification_job(p):
            raise ValueError('Job differs from the frozen qualification specification')
    else:
        raise ValueError('Invalid native role')


def validate_jobs(p):
    d = p['design']; variant = p['variant']
    expected = [dict(run_id=f'e{e}-h{h}-g{g:g}-s{s}', environment=e, heads=h,
        control=g, seed=s, steps=d['steps'], shared_seed=d['shared_seed'],
        save_optimizer=(variant['preparation'] != 'base' or s in d['seeds'][:2]))
        for e, g, h, s in product(d['environments'], d['controls'], d['heads'], d['seeds'])]
    jobs = p.get('jobs')
    if not isinstance(jobs, list) or len(jobs) != len(expected):
        raise ValueError('Frozen jobs do not cover the design exactly once')
    if any(not isinstance(j, dict) or type(j.get('save_optimizer')) is not bool or
           any(type(j.get(k)) is not int for k in ('environment','heads','seed','steps','shared_seed')) or
           type(j.get('control')) not in (int, float) for j in jobs):
        raise ValueError('Malformed frozen job')
    if sorted(jobs, key=lambda j: str(j.get('run_id'))) != sorted(expected, key=lambda j: j['run_id']):
        raise ValueError('Frozen job identifiers, cells, seeds, horizon or checkpoint policy differ')
    q = p.get('qualification')
    expected_q = dict(steps=128, heads=max(d['heads']), control=2., full_state_replay=True, role='qualification')
    if variant['observer'] == 'shared':
        expected_q['uninstrumented_comparison'] = True
        observer = p['shared_paths']
        for key, val in dict(schema='native-shared-parameter-path-v1', parameter_selector='reslayerAs',
                             cadence=256, include_first_update=True, first_raw_and_clipped_gradient=True).items():
            if observer.get(key) != val or type(observer.get(key)) is not type(val):
                raise ValueError('Unsupported shared-path observer contract')
    if q != expected_q or any(type(q[k]) is not type(v) for k,v in expected_q.items()):
        raise ValueError('Qualification specification differs from the supported native route')
    if p.get('replay_contract') != contract():
        raise ValueError('Wrong replay contract or comparator identity')


def validate_execution(p, study):
    file_hash(study/'design.json', p.get('design_sha256'), 'design')
    if read_json(study/'design.json') != p['design']:
        raise ValueError('Frozen design content differs')
    fixed = dict(
        architecture=dict(depth=5, head_dimension=64, generator_width=170, generator_residual_units=8),
        optimizer=dict(generator_rate='0.0003*g', other_rate='0.0003*2/N', betas=[.9,.95],
                       epsilon=1e-8, weight_decay=.01, global_norm_clip=1., foreach=False, schedule='constant'),
        arithmetic=dict(dtype='float32', tf32=False, observation_reductions='float64', cpu_threads=2),
        memory_ceiling_bytes=22*1024**3)
    for key,val in fixed.items():
        if p.get(key) != val:
            raise ValueError('Unsupported native execution contract: '+key)
    steps=p['design']['steps']
    milestones={t for t in (0,256,512,1024,2048,4096,8192,16384,32768) if t<=steps}
    if p['variant']['preparation']=='clock':
        milestones|={t for t in (steps//4,steps//2,steps) if t%64==0}
    if p['variant']['observer']=='shared':
        milestones|=set(range(0,steps+1,512))|{steps}
    expected=dict(cadence=64,head_contexts=64,evaluation_contexts=128,
        full_vocabulary_contexts=8,row_denominator_floor=1e-30,milestone_steps=sorted(milestones))
    if p.get('observations')!=expected:
        raise ValueError('Observation schedule or schema differs')


def native_vocabulary(path):
    """Read the literal configuration default without importing native model code."""
    tree = ast.parse(Path(path).read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PldrllmConfig']
    if len(classes) != 1:
        raise ValueError('Unsupported native configuration class')
    functions = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == '__init__']
    if len(functions) != 1:
        raise ValueError('Unsupported native configuration initializer')
    args = functions[0].args
    defaults = dict(zip([n.arg for n in args.args][-len(args.defaults):], args.defaults))
    try:
        value = ast.literal_eval(defaults['vocab_size'])
    except (KeyError, ValueError) as exc:
        raise ValueError('Vocabulary must have an authenticated literal default') from exc
    if type(value) is not int or value <= 0:
        raise ValueError('Invalid vocabulary size')
    return value


def integer_array(a, shape, name, dtype=None):
    if a.shape != tuple(shape) or a.dtype.kind not in 'iu' or (dtype is not None and a.dtype != dtype):
        raise ValueError('Wrong selection/data shape or integer dtype: '+name)


def tokens_in_range(a, vocab, name):
    for start in range(0, len(a), 4096):
        block = a[start:start+4096]
        if block.size and (int(block.min()) < 0 or int(block.max()) >= vocab):
            raise ValueError('Token outside native vocabulary: '+name)


def validate_inputs(p, study, locations, spec, resolve):
    expected = dict(locations)
    if p['variant']['preparation'] != 'base':
        expected['selection_decision'] = study/'selection-decision.json'
        decision = p['selection']
        exact_keys(decision, ['search_analyses','hypotheses','primary_fields','held_out_axes','decision_rules'], 'selection decision')
        if any(not v for v in decision.values()):
            raise ValueError('Incomplete refinement selection')
        searches = digest_map(decision['search_analyses'], 'search analysis')
        expected.update({f'search_analysis_{i}': resolve(name) for i,name in enumerate(sorted(searches))})
    if p['variant']['preparation'] == 'clock':
        expected['family_specification'] = study/'family.json'
    roles = p.get('input_roles')
    exact_keys(roles, expected, 'input')
    if any(not isinstance(n,str) or not n for n in roles.values()):
        raise ValueError('Malformed input role path')
    actual = {r: resolve(n).resolve() for r,n in roles.items()}
    if len(set(actual.values())) != len(actual) or any(actual[r] != Path(n).resolve() for r,n in expected.items()):
        raise ValueError('Misidentified or ambiguous input role')
    digests = digest_map(p.get('input_sha256'), 'input')
    # Resolve complete locations, never basenames, and reject aliases of the same file.
    resolved = {resolve(n).resolve(): v for n,v in digests.items()}
    if len(resolved) != len(digests) or set(resolved) != set(actual.values()):
        raise ValueError('Input digest roles are incomplete or unregistered')
    for path,digest in resolved.items():
        file_hash(path, digest, 'input')
    cm, pm = (read_json(actual[k]) for k in ('corpus_manifest','probe_manifest'))
    if cm.get('schema') != 'onepass-refinedweb-corpus-v1' or cm.get('status') != 'complete' or cm.get('training_documents') != spec.documents or cm.get('shape') != [spec.documents,spec.columns] or cm.get('dtype') != 'int32':
        raise ValueError('Corpus manifest schema or dimensions differ')
    if pm.get('schema') != 'controlled-cohort-v1' or pm.get('kind') != 'short' or pm.get('count') != spec.probe_documents or pm.get('length') != spec.columns:
        raise ValueError('Probe manifest schema or dimensions differ')
    for manifest,prefix,keys in [(cm,'corpus',('tokens','records')), (pm,'probe',('tokens','offsets','records'))]:
        for key in keys:
            if manifest.get(key+'_sha256') != resolved[actual[prefix+'_'+key]]:
                raise ValueError('Manifest does not bind '+prefix+'_'+key)
    records = read_json(actual['corpus_records']); probes = read_json(actual['probe_records'])
    if not isinstance(records,list) or len(records) != spec.documents or not isinstance(probes,list) or len(probes) != spec.probe_documents:
        raise ValueError('Document record count differs')
    hashes=[]; probe_hashes=[]
    for source,out in [(records,hashes),(probes,probe_hashes)]:
        for row in source:
            h = row.get('content_sha256') if isinstance(row,dict) else None
            if not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h):
                raise ValueError('Invalid document identity')
            out.append(h)
        if len(set(out)) != len(out):
            raise ValueError('Repeated document identity')
    if set(hashes) & set(probe_hashes[spec.probe_start:spec.probe_start+spec.probe_count]):
        raise ValueError('Evaluation document overlaps training')
    if p['variant']['preparation'] != 'base':
        if read_json(actual['selection_decision']) != p['selection']:
            raise ValueError('Selection decision content differs')
        file_hash(actual['selection_decision'], p.get('selection_record_sha256'), 'selection decision')
        seeds=set()
        for i,name in enumerate(sorted(p['selection']['search_analyses'])):
            path=actual[f'search_analysis_{i}'];file_hash(path,p['selection']['search_analyses'][name],'search analysis')
            record=read_json(path)
            if record.get('status') != 'complete' or not isinstance(record.get('design'),dict):
                raise ValueError('Uncompleted search selection')
            seeds.update(record['design']['seeds'])
        if not seeds or seeds & set(p['design']['seeds']):
            raise ValueError('Refinement initialization is not independent of selection')
    if p['variant']['preparation'] == 'clock':
        if read_json(actual['family_specification']) != p['joint_family']['specification']:
            raise ValueError('Family specification differs')
        file_hash(actual['family_specification'],p['joint_family'].get('specification_sha256'),'family specification')
    return actual


def validate_selection(p, study, paths, spec):
    digest_map({'selection':p.get('selection_sha256')}, 'selection')
    file_hash(study/'selection.npz',p['selection_sha256'],'selection')
    d=p['design'];source=p['source']
    expected_source=dict(master_documents=spec.documents,batch=spec.batch,prefix_tokens=spec.prefix,
        target_tokens=1,no_repeated_supervised_positions=True,partition_seed=9152501,stream_seed='9152502+environment')
    for key,val in expected_source.items():
        if source.get(key) != val or type(source.get(key)) is not type(val):
            raise ValueError('Unsupported source law: '+key)
    population_size=spec.documents//2
    if p['variant']['preparation'] == 'clock':
        family=p['joint_family'];f=family['specification']
        exact_keys(f,['name','heads','rate_ratios','seeds','environments','steps_per_head','corpus_documents_per_head','shared_seed'],'family')
        if len(d['heads'])!=1 or d['heads'][0] not in f['heads']:
            raise ValueError('Clock family width differs')
        n=d['heads'][0]
        if type(f['corpus_documents_per_head']) is not int or f['corpus_documents_per_head']<=0 or type(f['steps_per_head']) is not int or f['steps_per_head']<=0:
            raise ValueError('Invalid clock resource coefficients')
        population_size=n*f['corpus_documents_per_head']
        if source.get('population_policy')!='nested-width-prefix-v1' or d['steps']!=n*f['steps_per_head'] or d['controls']!=[2*r/n for r in f['rate_ratios']]:
            raise ValueError('Clock family horizon, rates or population law differ')
        if any(d[k]!=f[k] for k in ('seeds','environments','shared_seed')) or d['name']!=f['name']+f'-h{n}':
            raise ValueError('Clock family design differs')
        for key,val in dict(terminal_other_clock=.0006*f['steps_per_head'],terminal_generator_clocks=[.0006*f['steps_per_head']*r for r in f['rate_ratios']],consumed_fraction=spec.batch*d['steps']/(spec.segments*population_size)).items():
            if family.get(key)!=val:
                raise ValueError('Clock family resource or clock mismatch')
    elif source.get('population_policy','full-master-half-v1')!='full-master-half-v1':
        raise ValueError('Unsupported source population policy')
    if not 1<=population_size<=spec.documents//2 or source.get('population_documents')!=population_size or source.get('population_blocks')!=spec.segments*population_size or spec.batch*d['steps']>spec.segments*population_size:
        raise ValueError('Invalid population size or exhausted single-pass stream')
    vocab=native_vocabulary(paths['native_config'])
    master=np.load(paths['corpus_tokens'],mmap_mode='r',allow_pickle=False)
    probe=np.load(paths['probe_tokens'],mmap_mode='r',allow_pickle=False)
    offsets=np.load(paths['probe_offsets'],allow_pickle=False)
    integer_array(master,(spec.documents,spec.columns),'master',np.dtype('int32'))
    integer_array(probe,(spec.probe_documents,spec.columns),'probe',np.dtype('int32'))
    integer_array(offsets,(spec.probe_documents,),'offsets')
    if np.any(offsets<0) or np.any(offsets>=spec.columns-spec.prefix):
        raise ValueError('Invalid probe offsets')
    tokens_in_range(master,vocab,'master');tokens_in_range(probe,vocab,'probe')
    halves=np.random.default_rng(9152501).permutation(spec.documents).reshape(2,-1)
    try:
        with np.load(study/'selection.npz',allow_pickle=False) as selected:
            keys={'probe_rows','probes'} | {f'{name}_{e}' for e in d['environments'] for name in ('population','blocks')}
            if set(selected.files)!=keys or len(selected.files)!=len(keys):
                raise ValueError('Selection keys differ from declared environments')
            pr=selected['probe_rows'];pv=selected['probes']
            integer_array(pr,(spec.probe_count,),'probe_rows')
            integer_array(pv,(spec.probe_count,spec.prefix+1),'probes',np.dtype('int32'))
            if not np.array_equal(pr,np.arange(spec.probe_start,spec.probe_start+spec.probe_count)):
                raise ValueError('Probe row selection differs')
            if not np.array_equal(pv,probe[pr[:,None],offsets[pr,None]+np.arange(spec.prefix+1)]):
                raise ValueError('Probe tokens differ from authenticated rows and offsets')
            for e in d['environments']:
                pop=selected[f'population_{e}'];blocks=selected[f'blocks_{e}']
                integer_array(pop,(population_size,),'population')
                integer_array(blocks,(d['steps'],spec.batch),'blocks')
                expected=halves[e,:population_size]
                if not np.array_equal(pop,expected):
                    raise ValueError('Population differs from declared partition law')
                local=np.random.default_rng(9152502+e).permutation(population_size*spec.segments)[:spec.batch*d['steps']]
                expected_blocks=(spec.segments*expected[local//spec.segments]+local%spec.segments).reshape(d['steps'],spec.batch)
                if len(np.unique(blocks))!=blocks.size or not np.array_equal(blocks,expected_blocks):
                    raise ValueError('Repeated or inconsistent selected blocks')
    except (OSError,TypeError,EOFError,zipfile.BadZipFile) as exc:
        raise ValueError('Malformed selection archive') from exc


def validate_qualification(p, study, spec, resolve):
    root=study/'qualification';q=read_json(root/'verification.json');ph=sha256(study/'protocol.json')
    steps=p['qualification']['steps'];job=qualification_job(p)
    if q.get('status')!='passed' or q.get('protocol_sha256')!=ph or q.get('scientific_updates')!=0 or q.get('qualification_updates')!=2*steps or q.get('full_state_bitwise_replay') is not True or q.get('replay_contract')!=contract():
        raise ValueError('Incomplete qualification record, accounting or replay contract')
    expected={root/side/name for side in ('native','replay') for name in BASE_ARTIFACTS}
    shared=p['variant']['observer']=='shared'
    if shared:
        expected|={root/'replay'/name for name in RECORDER_ARTIFACTS}
        if q.get('uninstrumented_vs_instrumented_bitwise') is not True or q.get('shared_endpoint_matches_state') is not True:
            raise ValueError('Missing shared observer qualification claims')
    checked=digest_map(q.get('checked_sha256'),'qualification artifact')
    resolved={resolve(name).resolve():digest for name,digest in checked.items()}
    if len(resolved)!=len(checked) or set(resolved)!={path.resolve() for path in expected}:
        raise ValueError('Qualification artifact roles are incomplete or unregistered')
    for path,digest in resolved.items():file_hash(path,digest,'qualification artifact')
    states=[]
    with np.load(study/'selection.npz',allow_pickle=False) as selected:
        blocks=selected[f'blocks_{job["environment"]}'][:steps]
    for side in ('native','replay'):
        path=root/side;m=read_json(path/'manifest.json')
        for key,val in dict(status='complete',role='qualification',job=job,protocol_sha256=ph,
            producer_sha256=p['source_sha256'],completed_steps=steps,attempted_steps=steps,intended_steps=steps,
            scientific_updates=0,qualification_updates=steps,optimizer_saved=True,
            checkpoint_available=True,checkpoint_resumable=True,optimizer_partial_mutation=False,
            original_failure=None,native_state_contract=p['native_state_contract']).items():
            if key not in m or canonical_digest(m[key])!=canonical_digest(val):
                raise ValueError('Incomplete qualification manifest: '+key)
        if not isinstance(m.get('device'),str) or re.fullmatch(r'cuda:[0-9]+',m['device']) is None:
            raise ValueError('Qualification RNG device role must be a single CUDA worker')
        artifacts=digest_map(m.get('artifacts'),'manifest artifact')
        names={'final-state.pt','observations.npz'} | (set(RECORDER_ARTIFACTS) if shared and side=='replay' else set())
        exact_keys(artifacts,names,'manifest artifact')
        for name,digest in artifacts.items():
            if digest!=resolved[(path/name).resolve()]:raise ValueError('Qualification manifest digest differs')
        state=torch.load(path/'final-state.pt',map_location='cpu',weights_only=True)
        exact_keys(state,['model','optimizer','step','job','protocol_sha256','cpu_rng','cuda_rng'],'full checkpoint')
        if canonical_digest(state['job'])!=canonical_digest(job) or state['protocol_sha256']!=ph:
            raise ValueError('Checkpoint job/protocol binding differs')
        validate_checkpoint(state,job['heads'],job['control'],steps)
        states.append(state)
        with np.load(path/'observations.npz',allow_pickle=False) as obs:
            times=np.arange(0,steps+1,64,dtype=np.int64)
            milestones=sorted((set(p['observations']['milestone_steps'])|{0,steps})&set(times))
            names={'steps','heads','nll_path','blocks','training_loss','gradient_norm'}|{f'{key}_{t}' for t in milestones for key in ('logits','evaluation_nll')}
            if set(obs.files)!=names:raise ValueError('Incomplete qualification observation schema')
            shapes={'steps':(len(times),),'heads':(len(times),64,5,job['heads'],4),
                'nll_path':(len(times),64),'blocks':(steps,spec.batch),'training_loss':(steps,),'gradient_norm':(steps,)}
            for t in milestones:
                shapes[f'logits_{t}']=(8,native_vocabulary_from_protocol(p))
                shapes[f'evaluation_nll_{t}']=(128,)
            for key,shape in shapes.items():
                a=obs[key]
                if a.shape!=shape or a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
                    raise ValueError('Invalid or nonfinite qualification observations: '+key)
            integer_array(obs['steps'], (len(times),), 'qualification steps', np.dtype('int64'))
            integer_array(obs['blocks'], (steps,spec.batch), 'qualification blocks', np.dtype('int64'))
            if not np.array_equal(obs['steps'],times) or not np.array_equal(obs['blocks'],blocks):
                raise ValueError('Qualification observations disagree with time or source selection')
            validate_observations(obs,times,milestones)
    require_replay(*states)
    with np.load(root/'native/observations.npz',allow_pickle=False) as a,np.load(root/'replay/observations.npz',allow_pickle=False) as b:
        require_observations(a,b)
    if shared:
        path=root/'replay';meta=read_json(path/'shared-metadata.json')
        if meta.get('job')!=job or meta.get('role')!='qualification' or meta.get('protocol_sha256')!=ph or meta.get('schema')!='native-shared-parameter-path-v1':
            raise ValueError('Shared metadata binding differs')
        names=meta.get('names');model=states[1]['model']
        if not isinstance(names,list) or not names or len(set(names))!=len(names) or set(names)!={n for n in model if 'reslayerAs' in n}:
            raise ValueError('Incomplete shared parameter selection')
        if meta.get('shapes')!=[list(model[n].shape) for n in names]:raise ValueError('Shared parameter shapes differ')
        expected=np.concatenate([model[n].reshape(-1).numpy() for n in names])
        values=np.load(path/'shared-parameters.npy',allow_pickle=False)
        times=np.load(path/'shared-steps.npy',allow_pickle=False)
        expected_times=sorted({0,1,steps}|set(range(256,steps+1,256)))
        if not np.array_equal(times,expected_times) or meta.get('steps')!=expected_times or values.shape!=(len(times),len(expected)) or values.dtype!=np.float32 or not np.isfinite(values).all() or meta.get('shared_parameter_count')!=len(expected):
            raise ValueError('Incomplete shared recorder trajectory')
        require_replay(values[-1],expected)
        with np.load(path/'first-gradient.npz',allow_pickle=False) as gradients:
            if set(gradients.files)!={'raw_gradient','clipped_gradient'} or any(gradients[k].shape!=expected.shape or gradients[k].dtype!=np.float32 or not np.isfinite(gradients[k]).all() for k in gradients.files):
                raise ValueError('Incomplete or nonfinite first-gradient recorder')


def native_vocabulary_from_protocol(p):
    # Set by the CPU asset validator, never accepted from protocol JSON.
    return p['_validated_vocabulary']


def validate_protocol(p, study, repo, locations, *, scientific=False, spec=PRODUCTION, resolve=Path):
    p.pop('_admission',None)
    p.pop('_validated_vocabulary',None)
    sources=digest_map(p.get('source_sha256'),'source')
    exact_keys(sources,registered_sources(p),'source')
    for name,digest in sources.items():file_hash(repo/name,digest,'producer')
    validate_jobs(p)
    validate_execution(p, study)
    paths=validate_inputs(p,study,locations,spec,resolve)
    validate_selection(p,study,paths,spec)
    validate_binding(p,paths,repo)
    p['_validated_vocabulary']=native_vocabulary(paths['native_config'])
    if scientific:validate_qualification(p,study,spec,resolve)
    p['_admission']=dict(structural_preflight='passed',
        qualification_evidence='passed' if scientific else 'not_requested',
        completed_scientific_acquisition=False, native_updates_executed=0,
        asset_contract='production' if spec==PRODUCTION else 'helper_fixture')
    return p

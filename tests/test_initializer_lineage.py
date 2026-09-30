"""Check native tensor support and the actual shape-aware normalizer."""
from pathlib import Path
import math
import sys
import pytest
import torch
R=Path(__file__).resolve().parents[1]
D=R/'companions/dynamics'
sys.path[:0]=[str(D),str(D/'vendor/model/src')]
from model_rg.training import TrainingModel
from model_rg.variance_family import normalize_variance_initialization

@pytest.mark.parametrize('heads',[2,4])
def test_actual_native_plga_uniform_support_and_rescaling(heads):
    torch.set_num_threads(2)
    model=TrainingModel(D/'vendor/native/PLDR-LLM-v51-SOC-110M-1',heads,91031,'cpu')
    tensors={n:p.detach().clone() for n,p in model.model.named_parameters()
             if 'plgatt_layer' in n and n.rsplit('.',1)[-1] in {'Wlst','pwlst','alst'}}
    assert len(tensors)==15
    for p in tensors.values():
        assert p.shape==(heads,64,64)
        assert float(p.abs().max())<=math.sqrt(6/(64*(64+heads)))+1e-8
        assert bool((p>0).any() and (p<0).any())
    normalize_variance_initialization(model)
    scale=math.sqrt((64+heads)/66)
    for n,p in model.model.named_parameters():
        if n in tensors:
            assert torch.equal(p,tensors[n]*scale)
            assert float(p.detach().abs().max())<=math.sqrt(6/(64*66))+1e-8
    # This is exact variance arithmetic, not an empirical RNG variance claim.
    assert math.isclose(scale**2*2/(64*(64+heads)),2/(64*66),rel_tol=2e-15)


def test_normalizer_rejects_incompatible_declared_head_count():
    model=TrainingModel(D/'vendor/native/PLDR-LLM-v51-SOC-110M-1',2,91031,'cpu')
    model.config.num_attention_heads=3
    with pytest.raises(ValueError,match='Unexpected PLGA tensor shape'):
        normalize_variance_initialization(model)

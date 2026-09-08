import pytest
import torch
from megatron_mini.distributed.axis import Axis
from megatron_mini.labs.checks import run_case
pytestmark=pytest.mark.exercise

@pytest.mark.parametrize('case',['residency','zero3','zero3_sac','offload_transfer',
    'offload_optimizer','offload_param','offload_activation','sac','offload_checkpoint'])
def test_v3_local_case(case):
    run_case(case,Axis(),torch.device('cpu'))

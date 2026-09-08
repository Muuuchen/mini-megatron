from dataclasses import dataclass
from megatron_mini.exercise import todo

@dataclass(frozen=True)
class Reduction:
    axis: object
    divisor: float=1.0

@dataclass(frozen=True)
class GradRole:
    reductions: tuple=()
    count_for_norm: bool=True
    kind: str='dense'  # dense / tp_shard / expert / fsdp_shard
    valid_numel: int | None=None  # local flat shard valid prefix, excluding padding


def finalize_gradients(named_parameters,roles,norm_axes,max_norm=0.0):
    """T49: 按显式roles归约/缩放，然后unique logical norm与裁剪，返回preclip norm。
    每个参数必须有role；SP norm/bias先TP SUM，dense再DP×CP平均；专家只在
    EDP×CP上合并，但等长local mean约定的分母仍是denseDP×CP。
    FSDP已经RS的grad不重复平均；norm统计则要包含每个owner的所有shards。
    count_for_norm仅决定范数计数，不决定是否缩放/更新该参数的副本。
    所有norm_axes SUM平方和，最后sqrt；按role.valid_numel剔除flat padding。不要 blindly WORLD norm。
    验收: grad_roles单元测试 + dense_tp_zero2/moe_ep_fsdp的连续update。
    """
    todo('T49')

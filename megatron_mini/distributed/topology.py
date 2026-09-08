"""坐标顺序固定为 [DP, PP, TP]，TP 是最快变化的维度。"""
from dataclasses import dataclass
from megatron_mini.exercise import todo


@dataclass(frozen=True)
class ParallelDims:
    tp: int = 1
    pp: int = 1
    dp: int = 1

    def __post_init__(self):
        if min(self.tp, self.pp, self.dp) < 1:
            raise ValueError("parallel dimensions must be positive")

    @property
    def world_size(self):
        return self.tp * self.pp * self.dp

    def rank(self, d: int, p: int, t: int) -> int:
        if not (0 <= d < self.dp and 0 <= p < self.pp and 0 <= t < self.tp):
            raise ValueError(f"invalid coordinates {(d,p,t)} for {self}")
        return (d * self.pp + p) * self.tp + t

    def coords(self, rank: int) -> tuple[int, int, int]:
        if not 0 <= rank < self.world_size:
            raise ValueError(f"invalid rank {rank}")
        return rank // (self.pp * self.tp), (rank // self.tp) % self.pp, rank % self.tp


@dataclass(frozen=True)
class GroupSpec:
    axis: str
    ranks: tuple[int, ...]


def plan_groups(dims: ParallelDims) -> list[GroupSpec]:
    """T02: 枚举所有组（包括 singleton），每个进程返回完全相同的列表。

    输出顺序必须是：所有 TP 组、所有 PP 组、所有 DP 组。
    每一类中固定坐标按自然升序枚举，组内变化坐标也升序。
    TP 固定(d,p)，PP 固定(d,t)，DP 固定(p,t)。
    不要只返回当前 rank 所属的组；new_group 的全局调用顺序必须一致。
    验收：tests/test_exercises.py::test_group_plan，8-rank 算术不需要 8 张卡。
    """
    todo("T02")
